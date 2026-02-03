import numpy as NP
import math as M
import os as OS
import json5 as JSON5 #JSON5 needed because of trailing commas
import json as JSON
#IMPULSES
#CREATE IDENITY - CREATE INSTANCE

project = {
    "known_attributes":[["x","y","z","xrot","yrot","zrot","xsc","ysc","zsc"],["spr","ind","col","a"]],
    "textures":[],
    "sprite_names":[],
    "sprites":[],
    "nodes":[],
    "objects":{"-1":[]}, #(if nodeindex #0 has multiple objects) -> 0.8, 0.9, 0.10, 0.11 
    "object_nodes":{"-1":None},
    "animations":{}
}
misc = {
    "projectmode":"off",
    "nodestack":[],
    "objectstack":["."],
    "objectaddstack":"",
    "unusednodes":[], # Nodes that don't draw and don't stack
    "animcurrent":-1,
}



## MATRIX MULTIPLICATION matr1@matr2: Child @ Parent
## ORDER OF MATRIX MULTIPLICATION: ROTATION -> SCALING -> POSITION

# RANDOMNESS, ROUGHNESS, USING ROUNDING (FLOOR, CEILING) FOR REALISM

# (TIME) (EVENT) ["mat", ...] ["a","x",val,eas] ["pr",node]
# "0"
# "0":[0,0,0,0,0,0,1,1,1]

# A VISUALIZER, like an DAW arrangement

## INITIALIZATION (creating nodes, parenting, textures, and sprites)
## ANIMATION (creating animations, NO CHANGING PARENTS)


## ATTRIBUTE MULTIPLIER 

## ANIMATION RELATIVITY - WHEN AN ANIMATION STOPS, EVERYTHING POSITIONING WOULD CONTINUE WHERE IT LEFT OFF



######################################################
def matrix_index_rotation(rot,inds):
    ret = NP.eye(4)
    ang = M.radians(rot)
    ret[inds[0]//4,inds[0]%4] = M.cos(ang)
    ret[inds[1]//4,inds[1]%4] = M.sin(ang)
    ret[inds[2]//4,inds[2]%4] = -M.sin(ang)
    ret[inds[3]//4,inds[3]%4] = M.cos(ang)
    return ret

def matrix_build(x,y,z,xrot,yrot,zrot,xsc,ysc,zsc):
    ret = (
        matrix_index_rotation(yrot,(0,2,8,10))@
        matrix_index_rotation(xrot,(5,9,6,10))@
        matrix_index_rotation(zrot,(0,4,1,5))@
        NP.array([[xsc,0,0,0],[0,ysc,0,0],[0,0,zsc,0],[0,0,0,1]])
    )
    ret[3,0:3] = [x,y,z]
    return ret

def matrix_multiply(*mats):
    ret = NP.eye(4)
    for i in mats:
        ret @= i
    return ret

def matrix_build_identity():
    return NP.eye(4)

def matrix_inverse(mat):
    return NP.linalg.inv(mat)

def matrix_transform_vertex(mat,*vector):
    return (vector if len(vector)==4 else vector+(1,))@mat

def matrix_get_position(mat):
    return mat[3,0:3]

def matrix_1x16(mat):
    return mat.reshape(1,16)[0]

def matrix_4x4(mat):
    return mat.reshape(4,4)

####################################################

class ProjectError(Exception):
    pass

def __pvm__(mode):
    '''project_valid_mode'''
    if misc["projectmode"] != mode:
        raise ProjectError(f"(Currently in ProjectMode \"{misc['projectmode']}\") This Function must be executed in ProjectMode \"{mode}\"")
    
def __pka__(attr):
    '''project_known_attribute'''
    if not (project["known_attributes"][0].__contains__(attr) or project["known_attributes"][1].__contains__(attr)):
        raise ProjectError(f"\"{attr}\" : Unrecognized Attribute (Not in Lists {project['known_attributes']})")

    
class InitCreationRegion:
    def __enter__(self):
        __pvm__("off")
        misc["projectmode"] = "init"

    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["projectmode"] = "init(done)"

class AnimationCreationRegion:
    def __enter__(self):
        __pvm__("init(done)")
        misc["projectmode"] = "anim"

    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["projectmode"] = "anim(done)"

    
####################################################

def sprite_get_from_gamemaker(yyfile:str):
    __pvm__("init")
    # v2 GameMaker Sprite
    file = open(yyfile)
    data : dict = JSON5.parse(file.read())[0]
    file.close()
    dir = OS.path.dirname(yyfile)
    texs: list = project["textures"]
    sprtexs = []
    for i in data["frames"]:
        imgfile = dir+"/"+i["name"]+".png"
        if not texs.__contains__(imgfile):
            texs.append(imgfile)
        sprtexs.append(texs.index(imgfile))

    name = data["name"]
    if project["sprite_names"].__contains__(name):
        ProjectError(f"\"{name}\" : Sprite Already Exists")
    project["sprite_names"].append(name)
    project["sprites"].append(sprtexs)

    return {
        "name": name,
        "bbox":(data["bbox_left"],data["bbox_top"],data["bbox_right"],data["bbox_bottom"]),
        "origin":(data["sequence"]["xorigin"],data["sequence"]["yorigin"]),
        "size":(data["width"],data["height"])
    }
    

##########################################################

# Not used in a node

class _NodeIndex(int):
    pass

## nodestack array is backwards (when it comes to matrix mult order) -> [..., grandparent, parent, child]
## because of nodestack - a child of a node will always have an index greater than the parent - ()
class NodeStack(_NodeIndex):
    def __enter__(self):
        __pvm__("init")
        nodeindex = _NodeIndex(self.__int__())
        if misc["nodestack"].__contains__(nodeindex):
            raise ProjectError(f"(NodeStacks) \"Node #{nodeindex}\" : This Node already appears in a NodeStack")
        misc["nodestack"].append(nodeindex)
        if misc["objectaddstack"] != "":
            misc["objectstack"].append(misc["objectaddstack"])
        if misc["unusednodes"].__contains__(nodeindex):
            misc["unusednodes"].remove(nodeindex)
        return nodeindex
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["nodestack"].pop()
        if project["object_nodes"][misc["objectstack"][-1]] == self.__int__():
            misc["objectstack"].pop()


def node_create(draw=False,object=False,objectname:str=None,**attrs):
    __pvm__("init")
    
    node = {}
    matb = [0,0,0,0,0,0,1,1,1]
    mch = False

    for attr,val in attrs.items():
        __pka__(attr)
        if project["known_attributes"][0].__contains__(attr):
            matb[project["known_attributes"][0].index(attr)] = val
            mch = True
        elif project["known_attributes"][1].__contains__(attr):
            node[attr] = val
    if draw: 
        node["ns"] = misc["nodestack"].copy()
    if mch:
        node["matb"] = tuple(matb)
    project["nodes"].append(node)
    nodeindex = _NodeIndex(len(project["nodes"])-1)

    if not draw: misc["unusednodes"].append(nodeindex)
    if object:
        #if objectname:
        name = "."+objectname
        if project["objects"].__contains__(object): 
            raise ProjectError(f"(Objects) : Object \"{object}\" already exists")
        project["objects"][object] = [nodeindex] if draw else []
        project["object_nodes"][object] = nodeindex
    elif draw:
        project["objects"][misc["objectstack"][-1]].append(nodeindex) 
    misc["objectaddstack"] = object
    
    return nodeindex

def nodes_create_with_sprite_basic(spritename:str,origin:tuple,object:str=""):
    __pvm__("init")
    
    with NodeStack(node_create(object=object)) as pos:
        with NodeStack(node_create(x=-origin[0],y=-origin[1])) as org:
            sprinds = project["sprites"][project["sprite_names"].index(spritename)]
            inds = []
            for i in sprinds:
                inds.append(node_create(draw=True,spr=spritename,ind=i))
            node_create(spr=spritename,ind=i)
    return {
        "main":pos,
        "origin":org,
        "inds":inds
    }
## Use function recursion to add NodeStacks in loops
## [Pos] + [Scl] + [Rot + -Pos]


############################################################################

# ------------------------------------------------------------------------------------------ note! Attr mul must be added to the parent 

# (BEFORE AnimCreationRegion)
# Object node inds - Create instances (clones) of each object node ind 
# create an object FUNCTION (not CLASS) (node indices inside the function will be drawn)
# the object would return an index of the project object list, 
    # (an index as a list which includes the original drawn indices and their NS parent indices as well)

# (IN AnimCreationRegion)
# with animwrite(anim_create()) (new projectmode)
# with a(anim_key_index_create(x=3,y=5)) as n -> (creating as optional) Key Index (new projectmode)
# use underscore attr (eg. _x) as increment - and maybe (_x_) as multiplication

# (AFTER AnimCreationRegion)
# Assign Animation to an object instance by:
    # Assign Animation Key Indices to Object Node Indices 
    # multiple Object Node Inds can use one Anim Key Index




class _AnimationIndex(int):
    pass

class AnimationWrite(_AnimationIndex):
    def __enter__(self):
        __pvm__("anim")
        animindex = _AnimationIndex(self.__int__())
        misc["animcurrent"] = animindex
        misc["projectmode"] = "anim(write)"
        return animindex

    def __exit__(self,exc_type,exc_val,exc_tb):
        __pvm__("anim(write)")
        misc["animcurrent"] = -1
        misc["projectmode"] = "anim"

def animation_create():
    __pvm__("anim")
    project["animations"].append({"tracks":[],"len":0})
    ind = _AnimationIndex(len(project["animations"])-1)
    return ind



class _TrackIndex(int):
    pass

class TrackWrite(_TrackIndex):
    def __enter__(self):
        __pvm__("anim(write)")
        trackindex = _TrackIndex(self.__int__())
        misc["trackcurrent"] = trackindex
        misc["projectmode"] = "anim(track)"

    def __exit__(self,exc_type,exc_val,exc_tb):
        __pvm__("anim(track)")
        misc["trackcurrent"] = -1
        misc["projectmode"] = "anim(write)"


def animation_track_create(**attrs):
    __pvm__("anim(track)")
    for attr,val in attrs.items():
        __pka__(attr)




def animation_key_add(nodeindex:_NodeIndex,attr:str,val,delta:int,easing=1.0,func=0):
    __pvm__("anim(active)")
    __pka__(attr)
    d = max(int(delta),0)
    keys = project["animations"][-1][nodeindex][attr]
    if d == 0: keys.pop()
    keys.append([keys[-1][0]+d,val,easing,func])

'''
def anim_init_attr(nodeindex:_NodeIndex):
    __pvm__("anim(active)")
    anim = project["animations"][-1]
    anim["draw"].extend(nodeindices)
''' 

##########################
# obj: list(inds from nodes_create_with_sprite_basic)
# 


def done(mrasbasename:str=""):
    with open((mrasbasename if mrasbasename!="" else OS.path.splitext(__file__)[0])+".mras","w") as f: 
        f.write(JSON.encoder.JSONEncoder().encode(project))
        
    print(project)
    misc["projectmode"] = "done"



###################################################


'''
with InitCreationRegion():
    sprite_get_from_gamemaker("C:/Users/mikey/Documents/GameMakerStudio2/RomitronAnimation/sprites/Sprite6/Sprite6.yy")
    a = nodes_create_with_sprite_basic("Sprite6",(32,32),"Obj1")


    print(f"Unused Nodes (list of NodeIndices): {misc['unusednodes']}")
print(project)

with AnimationCreationRegion():
    anim_create(True)
    anim_enable_draw(*a["inds"])
    anim_key_add(a["main"],"xrot",100)
    anim_key_add(a["main"],"zrot",0)
    anim_key_add(a["main"],"yrot",0)
    anim_key_add(a["main"],"xsc",2)
    anim_key_add(a["main"],"ysc",2)
    anim_key_add(a["main"],"x",320)
    anim_key_add(a["main"],"y",180)(240)
    anim_key_add(a["main"],"xrot",180,0.2)
    anim_key_add(a["main"],"zrot",180,0.2)
    anim_key_add(a["main"],"yrot",180,0.1)

done()
'''

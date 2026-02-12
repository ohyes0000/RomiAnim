import numpy as NP
import math as M
import os as OS
import json5 as JSON5 #JSON5 needed because of trailing commas
import json as JSON
#IMPULSES
#CREATE IDENITY - CREATE INSTANCE

project = {
    "known_attributes":[["x","y","z","xrot","yrot","zrot","xsc","ysc","zsc"],["spr","ind","col","a"]],
    "separators":{"ns":"/"},
    "textures":[],
    "sprite_names":[],
    "sprites":[],
    "nodes":[],
    "node_data":[],
    "node_stack":[],
    "objects":[],
    "animations":[],
    
}
misc = {
    "time_value_func_args":(), 
    "projectmode":"off",
    "nodestacktop":-1,
    "nodeinitdatastring":[],
    "lastobjlastnode":-1,
    "unusednodes":[], # Nodes that don't draw and don't stack
    "unusednodedata":[],
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

def __pvm__(*mode):
    '''project_valid_mode'''
    if not mode.__contains__(misc["projectmode"]):
        raise ProjectError(f"(Currently in ProjectMode \"{misc['projectmode']}\") This Function must be executed in ProjectMode {list(mode)}")
    
def __pka__(attr):
    '''project_known_attribute'''
    if not (project["known_attributes"][0].__contains__(attr) or project["known_attributes"][1].__contains__(attr)):
        raise ProjectError(f"\"{attr}\" : Unrecognized Attribute (Not in Lists {project['known_attributes']})")

def __pie__(index,_list,name):
    '''project_index_exists'''
    if not (-1 < index < len(_list) > 0):
        raise ProjectError(f"({name}Index) : {name} Index #{index} doesn't exist")
    
def project_set_key_component_names(time:str,value:str,function:str,arguments:str):
    misc["time_value_func_args"] = (time,value,function,arguments)
    
class InitCreationRegion:
    def __enter__(self):
        __pvm__("off")
        project_set_key_component_names("delta","val","fn","args")
        misc["projectmode"] = "init"

    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["projectmode"] = "init(done)"

class AnimationCreationRegion:
    def __enter__(self):
        __pvm__("init(done)")
        misc["projectmode"] = "anim"

    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["projectmode"] = "anim(done)"

class SampleCreationRegion:
    def __enter__(self):
        __pvm__("anim(done)")
        misc["projectmode"] = "smpl"

    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["projectmode"] = "smpl(done)"  

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
        ProjectError(f"(Sprites) : \"{name}\" Sprite already exists")
    project["sprite_names"].append(name)
    project["sprites"].append(sprtexs)

    return {
        "name": name,
        "bbox":(data["bbox_left"],data["bbox_top"],data["bbox_right"],data["bbox_bottom"]),
        "origin":(data["sequence"]["xorigin"],data["sequence"]["yorigin"]),
        "size":(data["width"],data["height"])
    }

##########################################################


class _NodeDataIndex(int):
    def __init__(self,index):
        __pie__(self,project["node_data"],"NodeData")

class _NodeIndex(int):
    def __init__(self,index):
        __pie__(self,project["nodes"],"Node")


## nodestack array is backwards (when it comes to matrix mult order) -> [..., grandparent, parent, child]
## because of nodestack - a child of a node will always have an index greater than the parent - ()

class StartObject:
    def __enter__(self):
        __pvm__("init")
        misc["projectmode"] = "init(obj)"
    def __exit__(self,exc_type,exc_val,exc_tb):
        misc["projectmode"] = "init"
        project["objects"].append(len(project["nodes"])-1)

class NodeStack(_NodeIndex):
    def __init__(self,nodeindex:_NodeIndex):
        _NodeIndex(self) # __pie__

    def __stackin__(self):
        __pvm__("init(obj)")
        if len(project["objects"]) > 0:
            if project["objects"][-1] >= self:
                raise ProjectError(f"(NodeStacks) : Cannot add (Node #{self}) to this Object when it originates from a different Object")
            if (currnode := misc["nodestacktop"]) >= self:
                raise ProjectError(f"(NodeStacks) : No directional dependency -> Invalid (Current Node #{currnode} < New Node #{self})")
        misc["nodestacktop"] = self
        if misc["unusednodes"].__contains__(self):
            misc["unusednodes"].remove(self)

    def __stackout__(self):
        misc["nodestacktop"] = project["node_stack"][misc["nodestacktop"]]

    def __enter__(self):
        self.__stackin__()
        return self
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.__stackout__()

#
def node_data(**attrs):
    __pvm__("init","init(obj)")
    
    nodedata = {}
    matb = [0,0,0,0,0,0,1,1,1]
    mch = False

    for attr,val in attrs.items():
        __pka__(attr)
        if project["known_attributes"][0].__contains__(attr):
            matb[project["known_attributes"][0].index(attr)] = val
            mch = True
        elif project["known_attributes"][1].__contains__(attr):
            nodedata[attr] = val
    if mch:
        nodedata["matb"] = tuple(matb)

    ndstr = JSON.dumps(nodedata,sort_keys=True)
    if misc["nodeinitdatastring"].__contains__(ndstr):
        nodedataindex = misc["nodeinitdatastring"].index(ndstr)
    else:
        project["node_data"].append(nodedata)
        misc["nodeinitdatastring"].append(ndstr)
        nodedataindex = _NodeDataIndex(len(project["node_data"])-1)
        misc["unusednodedata"].append(nodedataindex)

    return nodedataindex

#
def node_create(nodedataindex:_NodeDataIndex=-1):
    __pvm__("init(obj)")
    if nodedataindex != -1:
        _NodeDataIndex(nodedataindex) # __pie__
        if misc["unusednodedata"].__contains__(nodedataindex):
            misc["unusednodedata"].remove(nodedataindex)
    project["node_stack"].append(misc["nodestacktop"])
    project["nodes"].append(nodedataindex)
    nodeindex = _NodeIndex(len(project["nodes"])-1)
    

    return nodeindex

#############
def nodes_create_with_sprite_basic(spritename:str,origin:tuple):
    with StartObject():
        with NodeStack(node_create()) as pos:
            with NodeStack(node_create(node_data(x=-origin[0],y=-origin[1]))) as org:
                sprinds = project["sprites"][project["sprite_names"].index(spritename)]
                inds = []
                for i in range(10):
                    inds.append(node_create(node_data(spr=spritename,ind=i)))
                     
    return {
        "main":pos,
        "origin":org,
        "inds":inds
    }
## Use function recursion to add NodeStacks in loops
## [Pos] + [Scl] + [Rot + -Pos]


############################################################################

# ------------------------------------------------------------------------------------------ note! Attr mul must be added to the parent 

class _AnimationIndex(int):
    def __init__(self,index):
        __pie__(self,project["animations"],"Animation")
        
    def __addkey__(self,init,attr:str,**comps):
        __pvm__("anim","anim(write)")
        # no __pka__
        key = []
        deford = misc["time_value_func_args"]
        for c in comps:
            if not deford.__contains__(c):
                raise ProjectError(f"(Animation) : Key Component \"{c}\" invalid")
        key.append(comps.setdefault(deford[0],0))
        key.append(comps.setdefault(deford[1],None))
        key.append(comps.setdefault(deford[2],0))
        key.append(comps.setdefault(deford[3],1))

        timeset = max(key[0],0)
        trk = project["animations"][self]["attrs"].setdefault(attr,[])

        if len(trk) > 0:
            addedtime = trk[-1][0]
            timeset += addedtime
            if timeset == addedtime: trk.pop()
        elif not init:
            raise ProjectError(f"(Animation) : Attribute \"{attr}\" was not initiated in this Animation")
        key[0] = timeset

        trk.append(tuple(key))
        return trk[-1]
        

class AnimationWrite(_AnimationIndex):
    def __init__(self,animationindex:_AnimationIndex):
        _AnimationIndex(self) # __pie__

    def __enter__(self):
        __pvm__("anim")
        misc["animcurrent"] = self
        misc["projectmode"] = "anim(write)"
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        __pvm__("anim(write)")
        misc["animcurrent"] = -1
        misc["projectmode"] = "anim"


def animation_create(**initattrs):
    __pvm__("anim")
    anim = {"attrs":{}}
    project["animations"].append(anim)
    animindex = _AnimationIndex(len(project["animations"])-1)

    for attr,val in initattrs.items():
        if attr[0] != "_": __pka__(attr)
        tvfa = misc["time_value_func_args"]
        animindex.__addkey__(True,attr,**{tvfa[0]:0,tvfa[1]:val})

    return animindex


def animation_key_add(attr,**comps):
    __pvm__("anim(write)")
    anim = misc["animcurrent"]
    if not isinstance(anim,_AnimationIndex):
        raise ProjectError("(Animation) : There is no active Animation to write in")
    if not project["animations"][anim]["attrs"].__contains__(attr):
        raise ProjectError(f"(Animation) : Must intialize Attribute \"{attr}\"")

    return anim.__addkey__(False,attr,**comps)[0]


##########################
# obj: list(inds from nodes_create_with_sprite_basic)
# 

#-----------------------------------------------------------
print("*1,*2,*3(*4,*5(*6)),*7".find("(",0,15))
def sample_create():
    pass

#-----------------------------------------------------------
def project_done(basename:str=""):
    filecontents = "{\n"
    keys = list(project.keys())
    for key in keys:
        filecontents += f"  \"{key}\": {JSON.dumps(project[key])}{'' if keys[-1]==key else ','}\n"
    filecontents += "}"

    open((basename if basename!="" else OS.path.splitext(__file__)[0])+".rasdat","w").write(filecontents)
    misc["projectmode"] = "done"



###################################################
# C:/Users/mikey/Documents/GameMakerStudio2/RomitronAnimation/sprites/Sprite6/Sprite6.yy
#

with InitCreationRegion():
    sprite_get_from_gamemaker("sTextBox.yy")
    a = nodes_create_with_sprite_basic("sTextBox",(32,32))
    print(f"Unused Nodes (list of NodeIndices): {misc['unusednodes']}")

with AnimationCreationRegion():
    with AnimationWrite(animation_create(x=None,y=None,_a=2)) as b:
        animation_key_add("_a",delta=3)


project_done()
# Create a sample using one object
# one track to one node (within object)
# underscored attribute tracks can be added or multiplied to multiple known_attributes


# After SampleCreationRegion
#One sample can be a child of another sample
# make project[node_list] that would have all the nodes taking an index from project[nodes]  
# create __gnv__ -> Get Node Variable
# do not allow the same node to be stacked twice
# create node_tags dict

# samplenodeindices

with SampleCreationRegion():
    pass

#print(project)



'''
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

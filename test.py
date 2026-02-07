import numpy as NP
import math as M
import os as OS
import json5 as JSON5 #JSON5 needed because of trailing commas
import json as JSON
#IMPULSES
#CREATE IDENITY - CREATE INSTANCE

project = {
    "known_attributes":[["x","y","z","xrot","yrot","zrot","xsc","ysc","zsc"],["spr","ind","col","a"]],
    "d_v_e_f":["delta","val","easing","func"], # although the "delta" key is "time", it must stay as "delta" as the argument of the "key_add" function
    "separators":{"defobj":"#", "obj":":", "ns":"/"},
    "key_components_order":["delta","val","func","func"],
    "textures":[],
    "sprite_names":[],
    "sprites":[],
    "nodes":[],
    "objects":{}, #(if nodeindex #0 has multiple objects) -> 0.8, 0.9, 0.10, 0.11 
    "animations":[],
    
}
misc = {
    "projectmode":"off",
    "nodestack":[],
    "objectstack":[],
    "objectaddstack":"",
    "nodeobjectcount":{},
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

def __pvm__(*mode):
    '''project_valid_mode'''
    if not mode.__contains__(misc["projectmode"]):
        raise ProjectError(f"(Currently in ProjectMode \"{misc['projectmode']}\") This Function must be executed in ProjectMode {list(mode)}")
    
def __pka__(attr):
    '''project_known_attribute'''
    if not (project["known_attributes"][0].__contains__(attr) or project["known_attributes"][1].__contains__(attr)):
        raise ProjectError(f"\"{attr}\" : Unrecognized Attribute (Not in Lists {project['known_attributes']})")

    
class InitCreationRegion:
    def __enter__(self):
        __pvm__("off")
        for c in project["d_v_e_f"]:
            if not project["key_components_order"].__contains__(c):
                raise ProjectError(f"(Internal) : There is no \"{c}\" Key Component") 
        misc["projectmode"] = "init"
        sep = project["separators"]["defobj"]
        project["objects"][sep] = []
        misc["objectstack"].append(sep)

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

class _NodeIndex(int):
    pass

## nodestack array is backwards (when it comes to matrix mult order) -> [..., grandparent, parent, child]
## because of nodestack - a child of a node will always have an index greater than the parent - ()
class NodeStack(_NodeIndex):
    def __init__(self,nodeindex:_NodeIndex):
        pass

    def __stackin__(self):
        __pvm__("init")
        if len(misc["nodestack"]) > 0:
            if not misc["nodestack"][-1] < self:
                raise ProjectError(f"(NodeStacks) : Node(#{self}) of Erred NodeStack must be created AFTER Node(#{misc["nodestack"][-1]}) of Current NodeStack")
        misc["nodestack"].append(self)
        if misc["unusednodes"].__contains__(self):
            misc["unusednodes"].remove(self)
    
    def __stackout__(self):
        __pvm__("init")
        misc["nodestack"].pop()

    def __enter__(self):
        self.__stackin__()
        return self
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.__stackout__()


class NodeStackObject(NodeStack):
    def __init__(self,nodeindex:_NodeIndex):
        count = misc["nodeobjectcount"].setdefault(self,0)
        self.__objname__ = project["separators"]["defobj"]+(f"{self}{project["separators"]["obj"]}{count}" if count>0 else f"{self}")
        misc["nodeobjectcount"][self]+=1
        project["objects"][self.__objname__] = []

    def __enter__(self):
        self.__stackin__()
        misc["objectstack"].append(self.__objname__)
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        self.__stackout__()
        misc["objectstack"].pop()

    def change_name(self,name:str):
        __pvm__("init")
        orgname = self.__objname__
        if orgname != name:
            if name == "":
                raise ProjectError("(Objects) : Object name cannot be blank")
            if name.startswith(project["separators"]["defobj"]):
                raise ProjectError(f"(Objects) : Object name cannot start with \"{project['separators']['defobj']}\"")
            if project["objects"].__contains__(name):
                raise ProjectError(f"(Objects) : Object name \"{name}\" already taken")
            project["objects"][name] = project["objects"].pop(self.__objname__)
            if misc["objectstack"].__contains__(orgname):
                ind = misc["objectstack"].index(orgname)
                misc["objectstack"].pop(ind)
                misc["objectstack"].insert(ind,name)
            self.__objname__ = name
        return self


def node_create(**attrs):
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
    if mch:
        node["matb"] = tuple(matb)
    project["nodes"].append(node)
    nodeindex = _NodeIndex(len(project["nodes"])-1)
    misc["unusednodes"].append(nodeindex)
    
    return nodeindex

def node_object_draw(nodeindex:_NodeIndex):
    __pvm__("init")
    obj = project["objects"][misc["objectstack"][-1]]
    if obj.__contains__(nodeindex):
        raise ProjectError(f"(Objects) : Node(#{nodeindex}) already has been drawn in this Object")
    obj.append(nodeindex)
    if misc["unusednodes"].__contains__(nodeindex):
        misc["unusednodes"].remove(nodeindex)
    node = project["nodes"][nodeindex]
    if len(misc["nodestack"]) > 0:
        sep = project["separators"]["ns"]
        node.setdefault("ns",[]).append("".join(str(ni)+sep for ni in misc["nodestack"]).strip(sep))

    return nodeindex

def nodes_create_with_sprite_basic(spritename:str,origin:tuple):
    __pvm__("init")

    with NodeStackObject(node_create()) as pos:
        with NodeStack(node_create(x=-origin[0],y=-origin[1])) as org:
            sprinds = project["sprites"][project["sprite_names"].index(spritename)]
            inds = []
            for i in sprinds:
                inds.append(node_object_draw(node_create(spr=spritename,ind=i)))
                     
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

# NO ANIMATION KEY INDICES - JUST ANIMATION INDICES
# AN ANIMATION CONTROLS ONLY ONE NODE ()
# Value (Null or Undefined) = Initial Node Value
class _AnimationIndex(int):
    def __addkey__(self,init,attr:str,**comps):
        __pvm__("anim","anim(write)")
        # no __pka__
        key = []
        order = project["key_components_order"]
        indextime = order.index(project["d_v_e_f"][0])
        for c in order:
            if not comps.__contains__(c):
                raise ProjectError(f"(Animation) : Key doesn't contain \"{c}\" component")
            key.append(comps[c])
        timeset = max(key[indextime],0)
        trk = project["animations"][self]["attrs"].setdefault(attr,[])

        if len(trk) > 0:
            addedtime = trk[-1][indextime]
            timeset += addedtime
            if timeset == addedtime: trk.pop()
        elif not init:
            raise ProjectError(f"(Animation) : Attribute \"{attr}\" was not initiated in this Animation")
         
        key[indextime] = timeset
        trk.append(tuple(key))
        return trk[-1]
        

class AnimationWrite(_AnimationIndex):
    def __init__(self,animationindex:_AnimationIndex):
        pass

    def __enter__(self):
        __pvm__("anim")
        misc["animcurrent"] = self
        misc["projectmode"] = "anim(write)"
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        __pvm__("anim(write)")
        misc["animcurrent"] = -1
        misc["projectmode"] = "anim"


## KEYS REPLACE INIT VALUES
## can make independent animations that take attr mul
# keys time must start at zero
def animation_create(**initattrs):
    __pvm__("anim")
    anim = {"attrs":{}}
    project["animations"].append(anim)
    animindex = _AnimationIndex(len(project["animations"])-1)

    for attr,val in initattrs.items():
        if attr[0] != "_": __pka__(attr)
        animindex.__addkey__(True,attr,val=val,delta=0,easing=1,func=0)

    return animindex


def animation_key_add(attr,**comps):
    __pvm__("anim(write)")
    anim = misc["animcurrent"]
    if not isinstance(anim,_AnimationIndex):
        raise ProjectError("(Animation) : There is no active Animation to write in")
    if not project["animations"][anim]["attrs"].__contains__(attr):
        raise ProjectError(f"(Animation) : Must intialize Attribute \"{attr}\"")

    order = project["d_v_e_f"]
    comps.setdefault(order[0],0)
    comps.setdefault(order[1],None)
    comps.setdefault(order[2],1)
    comps.setdefault(order[3],0)
    return anim.__addkey__(
        False,attr,**comps
    )[project["key_components_order"].index(project["d_v_e_f"][0])]


##########################
# obj: list(inds from nodes_create_with_sprite_basic)
# 

#-----------------------------------------------------------

def sample_create():
    pass

#-----------------------------------------------------------
def done(mrasbasename:str=""):
    with open((mrasbasename if mrasbasename!="" else OS.path.splitext(__file__)[0])+".mras","w") as f: 
        f.write(JSON.encoder.JSONEncoder().encode(project))
        
    print(project)
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
        animation_key_add(attr="_a",delta=3)


# Create a sample using one object
with SampleCreationRegion():
    pass

print(project)



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


# coding: utf-8

# In[2]:


import numpy as np


# In[11]:


def create_image_to_urls_resnet(img_dict, resnet_dict):
    """ Given the img dict and the resnet dictionary, 
    returns a dictionary that maps the image id to a tuple of the url and resnet feature.
    
    PARAMETERS
    ----------
    img_dict: list of image dictionaries (map "id" and "url" to those numbers), d["image"]
    resnet_dict: dictionary that maps image id to resnet feature
    
    RETURN
    ------
    out:  dictionary of 2 dictionaries that maps image id to url and to resnet feature
    """
    
    out = dict()
    #url_dict = dict()
    
    url_dict = {entry["id"]:entry["coco_url"] for entry in img_dict if entry["id"] in resnet_dict}
    
    out = {"urls":url_dict,"resnet":resnet_dict}
    return out
    


# In[12]:


"""
img_dict = ({"id":50,"coco_url":"yay"},{"id":10,"coco_url":"woot"},{"id":20,"coco_url":"yeote"})
resnet_dict = {50:(1,2,3,4,5,6),10:(4,5,6,7,8,9)}

create_image_to_resnet_dictionary(img_dict,resnet_dict)
"""



# In[18]:


def create_caption_id_to_string(capt_dict, resnet_dict):
    """Given the caption dictionary and resnet_dict, returns dictionary that maps caption id to its string
    
    PARAMETERS
    ---------
    capt_dict: list of caption dicts (map "id","image_id", and "caption" to those vals), d["annotations"]
    resnet_dict: dictionary that maps image id to resnet feature
    
    RETURN
    -----
    out: dictionary that maps caption id to the string
    """
    
    out = {entry["id"]:entry["caption"] for entry in capt_dict if entry["id"] in resnet_dict}
    
    return out


# In[27]:


"""
capt_dict = ({"id":50,"caption":"yay","image_id":5000},{"id":10,"caption":"woot","image_id":50001},{"id":20,"caption":"yeote","image_id":50000001})
resnet_dict = {50:(1,2,3,4,5,6),10:(4,5,6,7,8,9),20:(1,2,9000,4,5,6)}

capt_str = create_caption_id_to_string(capt_dict,resnet_dict)
"""


# In[28]:


def create_caption_id_to_embedding_string_img_id(capt_dict,embeddings,capt_id_string):
    """Given the caption dictionary, embedding dictionary,and capt_id_string dict, returns a dictionary of 3 dictionaries
        that maps the caption to its image id, string, and embeddings
    
    PARAMETERS
    ---------
    capt_dict: list of caption dicts (map "id","image_id", and "caption" to those vals), d["annotations"]
    embeddings: dictionary that maps caption id to embedding
    capt_id_string: dictionary of embeddings 
    
    RETURN
    -----
    out: dictionary of 3 dictionaries that maps caption id to image id,string,embeddings
    """
    
    out = dict()
    capt_to_id= {entry["id"]:entry["image_id"] for entry in capt_dict if entry["id"] in capt_id_string}
    out = {"img_id":capt_to_id,"semantics":embeddings,"string":capt_id_string}
    return out
    


# In[29]:


"""
embed = {5:(1,2,3,4,5),7:(1,2,3,4,9),10:(1,2,999,4,5)}
create_caption_id_to_embedding_string_img_id(capt_dict,embed,capt_str)
"""


from IPython.display import Image
from IPython.display import display

def display_images(ids, image_id_to_url):
    """ Displays images corresponding to the specified ids within 
    a jupyter notebook cell. 
    
        Parameters
        ----------
        ids : nd.array
            the ids corresponding to the images to be displayed
            
        image_id_to_url : dictionary
            maps image ids to url """
    imgs = []
    
    for i, id in enumerate(ids):
        imgs.append(Image(url = image_id_to_url[id]))
        
    display(*imgs)
    return
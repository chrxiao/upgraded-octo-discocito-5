def populate_image_id_to_semantic_database(model, normalize, image_id_to_features):
    """ Populates a database dictionary that maps image ids to their 
    normalized semantic features. 
    
        Parameters
        ----------
        model : function
            given a set of image features, return its semantic features
            
        normalize : function
            returns the normalized semantic features
            
        image_id_to_features : dictionary
            maps image ids to image features 
                    
        Returns
        -------
        A database dictionary that maps image ids to normalized semantic 
        features. """
    
    return {id:normalize(model(features)) for id, features in image_id_to_features.items()}
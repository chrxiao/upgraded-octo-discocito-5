def populate_id_to_semantic_database(model, id_to_features):
    """ Populates a database dictionary that maps image ids to their 
    semantic features. 
    
        Parameters
        ----------
        model : function
            given a set of image features, return its semantic features
            
        id_to_features : dictionary
            maps image ids to image features 
                    
        Returns
        -------
        A database dictionary that maps image ids to semantic features. """
    
    return {id:model(features) for id, features in id_to_features.items()}
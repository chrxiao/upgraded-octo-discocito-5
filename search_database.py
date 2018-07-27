from sklearn.metrics.pairwise import cosine_similarity

def search_database(query_vec, id_to_semantic_database, k):
    """ Given a shape (50,) representative vector for a query, return
    the top k image ids of images matching the query.
    
        Parameters
        ----------
        query_vec : nd.array
            
        id_to_semantic_database : dictionary
            maps image ids to semantic features
            
        k : scalar
            determines how many of the top image ids should be returned 
                    
        Returns
        -------
        An array of shape (k,) containing the image ids of the top k 
        images. """
    
    top_k_cos_sim = np.zeros(k)
    top_k_id = np.zeros(k)
    
    for id, semantic in id_to_semantic_database.items():
        cos_sim = cosine_similarity(query_vec.reshape(1, -1), semantic.reshape(1, -1))[0][0]
        if cos_sim > np.min(top_k_cos_sim):
            new_i = np.argmin(top_k_cos_sim)
            top_k_cos_sim[new_i] = cos_sim
            top_k_id[new_i] = id
            
    # sort in descending order 1 -> 0
    sort_i = np.argsort(top_k_cos_sim)[::-1]
    return top_k_id[sort_i]
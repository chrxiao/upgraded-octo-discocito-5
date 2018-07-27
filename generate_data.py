import numpy as np

def generate_good_bad(cap_to_img, imgs, K):
    """ Creates triples of captions to their corresponding good-images and random bad-images.
        Returns only ids, not actual data
    
    PARAMETERS
    ---------
    cap_to_img: dictionary of caption_ids to image_ids
                {int: int}
    
    imgs:       a collection of image_ids
                iterable
    
    K:          number of bad_img_ids to generate for each caption_id
                int
    
    RETURN
    ------
    out:        collection of N = C*K triples of caption_id, good_img_id, bad_img_id
                caption_id and good_img_id are the same for each K-sized batch of random bad_img_ids
                numpy.ndarry of shape (N, 3)"""
    cap_and_good = np.repeat(generate_good(cap_to_img), K, axis=0)
    bad = np.random.choice(imgs, (cap_and_good.shape[0], 1))
    return np.concatenate((cap_and_good, bad), axis=1)

def generate_good(cap_to_img):
    """ Creates pairs of captions to their corresponding good-images.
        Returns only ids, not actual data
    
    PARAMETERS
    ---------
    cap_to_img: dictionary of caption_ids to image_ids
                {int: int}
    
    RETURN
    ------
    out:        collection of N = C pairs of caption_id, good_img_id
                numpy.ndarry of shape (N, 2)"""
    return np.array(list(cap_to_img.items()))

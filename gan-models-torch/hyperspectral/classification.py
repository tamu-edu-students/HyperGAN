import numpy as np
import pysptools.distance as distance
import os
import math


class Objective:
    
    def __init__(self):
        None
    
    def SAM(self, ref, input):
        return distance.SAM(ref, input)
    
    def SID(self, ref, input):
        return distance.SID(ref, input)

    def SAD(self, ref, input):
    
        vector1 = np.asarray(ref)
        vector2 = np.asarray(input)

        dot_product = np.dot(vector1, vector2)

        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        spectral_angle = dot_product / (magnitude1 * magnitude2)

        spectral_angle = np.clip(spectral_angle, -1, 1)

        return np.arccos(spectral_angle)
    
    def SCM(self, ref, input):
        ref_spectrum = np.asarray(ref)
        pixel_spectrum = np.asarray(input)
        
        sum1, sum2, sum3, mean1, mean2 = 0, 0, 0, 0, 0
        num_layers = ref_spectrum.size
        
        mean1 = np.mean(pixel_spectrum)
        mean2 = np.mean(ref_spectrum)

        sum1 = np.sum((pixel_spectrum - mean1) * (ref_spectrum - mean2))
        sum2 = np.sum((pixel_spectrum - mean1) ** 2)
        sum3 = np.sum((ref_spectrum - mean2) ** 2)

        if sum2 <= 0 or sum3 <= 0:
            return -1  # set to white due to an error

        scm_val = sum1 / math.sqrt(sum2 * sum3)

        return scm_val*-1
    
    def EUD(self, ref, input):
        ref_spectrum = np.asarray(ref)
        pixel_spectrum = np.asarray(input)
        
        sum_squared_diff = np.sum((ref_spectrum - pixel_spectrum) ** 2)
        
        return np.sqrt(sum_squared_diff)



class Classify:

    def __init__(self, evaluation='SAD') -> None:
        
        objective = Objective()
        self.evaluator = None

        if evaluation == 'SAD':
            self.evaluator = objective.SAD
        if evaluation == 'SID':
            self.evaluator = objective.SID
        if evaluation == 'SAM':
            self.evaluator = objective.SAM
        if evaluation == 'SCM':
            self.evaluator = objective.SCM
        if evaluation == 'EUD':
            self.evaluator = objective.EUD
    
    def classify_by_min(self, ref_list, input):

        """
        This function classifies a pixel given the reference endmembers
        inputs:
            ref_list = reference endmembers as a numpy array N x B, where N is the number of endmembers and B is the number of bands
            input = pixel to classify, numpy array of dimension B
            evaluation = SAD, SAM, SID
        outputs:
            index of min element
        """
        scores = {}
        for label, val in ref_list.items():
            scores[label] = self.evaluator(val, input) 
        
        return min(scores, key=lambda k: scores[k])
    
    def compute_divergence_measure(self, target, ref):
        return self.evaluator(target, ref)
    


        
    
    





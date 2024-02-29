import numpy as np
import pysptools.distance as distance
import os
import math
import torch


class Objective:
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_gpu = True if torch.cuda.is_available() else False
    
    def SAM(self, ref, input):

        if self.use_gpu:
            try:
                s1_tensor = torch.tensor(ref, dtype=torch.float32).to(self.device)
                s2_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
                s1_norm = torch.norm(s1_tensor)
                s2_norm = torch.norm(s2_tensor)
                sum_s1_s2 = torch.dot(s1_tensor, s2_tensor)
                angle = torch.acos(sum_s1_s2 / (s1_norm * s2_norm)).cpu().numpy()
            except ValueError:
                # PyTorch doesn't like when acos is called with
                # a value very near to 1
                return 0.0
            return angle.item()
        else:
            return distance.SAM(ref, input)


    def SID(self, ref, input):
        
        if self.use_gpu:
            s1 = torch.tensor(ref).to(self.device)
            s2 = torch.tensor(input).to(self.device)
            p = (s1 / torch.sum(s1)) + np.spacing(1)
            q = (s2 / torch.sum(s2)) + np.spacing(1)
            return torch.sum(p * torch.log(p / q) + q * torch.log(q / p)).item()
        else:
            return distance.SID(ref, input)

    def SAD(self, ref, input):
        if self.use_gpu:
            ref_tensor = torch.tensor(ref, dtype=torch.float32).to(self.device)
            input_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
            dot_product = torch.dot(ref_tensor, input_tensor)
            magnitude_ref = torch.norm(ref_tensor)
            magnitude_input = torch.norm(input_tensor)
            spectral_angle = dot_product / (magnitude_ref * magnitude_input)
            spectral_angle = torch.clamp(spectral_angle, -1, 1)  # Clip to valid range
            return torch.acos(spectral_angle).item()
        else:
            vector1 = np.asarray(ref)
            vector2 = np.asarray(input)
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            spectral_angle = dot_product / (magnitude1 * magnitude2)
            spectral_angle = np.clip(spectral_angle, -1, 1)
            return np.arccos(spectral_angle)        


    def SCM(self, ref, input, to_min=False):
        if self.use_gpu:
            ref_spectrum = torch.tensor(ref, dtype=torch.float32)
            pixel_spectrum = torch.tensor(input, dtype=torch.float32)
            mean1 = torch.mean(pixel_spectrum)
            mean2 = torch.mean(ref_spectrum)
            sum1 = torch.sum((pixel_spectrum - mean1) * (ref_spectrum - mean2))
            sum2 = torch.sum((pixel_spectrum - mean1) ** 2)
            sum3 = torch.sum((ref_spectrum - mean2) ** 2)

            if sum2 <= 0 or sum3 <= 0:
                return -1  # set to white due to an error

            scm_val = (sum1 / torch.sqrt(sum2 * sum3)).item()
            
        else:
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

        if to_min:
            return scm_val*-1
        else:
            return scm_val
    
    def EUD(self, ref, input):
        
        if self.use_gpu:
            ref_spectrum = torch.tensor(ref, dtype=torch.float32)
            pixel_spectrum = torch.tensor(input, dtype=torch.float32)
            
            sum_squared_diff = torch.sum((ref_spectrum - pixel_spectrum) ** 2)

            return torch.sqrt(sum_squared_diff).item()
        else:
            ref_spectrum = np.asarray(ref)
            pixel_spectrum = np.asarray(input)
            
            sum_squared_diff = np.sum((ref_spectrum - pixel_spectrum) ** 2)
            
            return np.sqrt(sum_squared_diff)


class Classify:

    def __init__(self, evaluation='SAD') -> None:
        
        objective = Objective()
        self.evaluation = evaluation
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

    def find_best_patch(self, orig, replacement_patches, replacement_locations):
        
        assert len(replacement_patches) == len(replacement_locations), "Arrays are not the same size"

        best_candidate = None
        best_candidate_loc = None
        best_score = np.iinfo(np.uint8).max

        for i in range(len(replacement_patches)):
            curr_score = self.evaluator(orig, replacement_patches[i], to_min=True) 
            if curr_score < best_score:
                best_candidate = replacement_patches[i]
                best_candidate_loc = replacement_locations[i]
                best_score = curr_score
        
        return best_candidate, best_candidate_loc
        

    


        
    
    





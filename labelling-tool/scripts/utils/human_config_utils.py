import toml
from .logging_utils import log_info, log_warning

class HumanConfigUtils():
    def __init__(self):
        self.human_traj_file = None

    def start_human_traj(self):
        self.dict = self.load_human_config(self.human_traj_file)
        self.used_indices = [int(key.replace("human", "")) for key in self.dict.keys()]

    def load_human_config(self, path):
        try:
            with open(path, "r") as f:
                return toml.load(f)
        except FileNotFoundError:
            log_warning(f"Human Config file {path} not found. Using an empty dict")
            return {}
    
    def save_human_config(self, path):
        with open(path, 'w') as f:
            toml.dump(self.dict, f)
            log_info(f"Config file was saved to {path}.")
            
    def __len__(self):
        return len(self.dict)
    
    def get_newID(self):
        used_set = set(self.used_indices)
        new_ID = 0
        
        while new_ID in used_set:
            new_ID += 1
        return new_ID

    def newID_init(self):
        newHuman_ID = self.get_newID()
        self.used_indices.append(newHuman_ID)
        newHuman_name = f"human{newHuman_ID}"
        
        self.dict[newHuman_name] = {}
        self.dict[newHuman_name]["traj_start"] = None
        self.dict[newHuman_name]["trajectories"] = None
        self.dict[newHuman_name]["human_context"] = None
        
        return newHuman_ID
    
    def delete_ID(self, humanID):
        if f"human{humanID}" in self.dict:
            self.dict.pop(f"human{humanID}", None)
            log_info(f"Deleted trajectory ID: {humanID}")
        else:
            log_warning(f"Tried to delete non-existent humanID: {humanID}")
        
        if humanID in self.used_indices:
            self.used_indices.remove(humanID)

    def get_element(self, humanID, elementName):
        if humanID in self.used_indices:
            return self.dict[f"human{humanID}"].get(elementName, None) 
        return None

    def set_element(self, humanID, elementName, value):
        if f"human{humanID}" not in self.dict:
            log_warning(f"Creating new entry for missing humanID: {humanID}")
            self.dict[f"humanID{humanID}"] = {}
        self.dict[f"human{humanID}"][elementName] = value
    
    def exist(self, humanID):
        return humanID in self.used_indices
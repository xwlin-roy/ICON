"""
Prompt Recorder Module
Records all prompts used during the attack process for each sample
"""
from typing import Dict, Any, Optional, List


class PromptRecorder:
    """
    Records all prompts used during the attack process
    """
    
    def __init__(self):
        """Initialize prompt recorder"""
        self.prompts: Dict[str, Any] = {}
        self.reset()
    
    def reset(self):
        """Reset recorder for a new sample"""
        self.prompts = {
            'first_attack': {
                'turn1_prompt': None,
                'turn2_prompt': None,
                'turn3_prompt': None
            },
            'second_attack': {
                'turn1_prompt': None,
                'turn2_prompt': None,
                'turn3_prompt': None
            },
            'tactical_optimizations': [],  # List of dicts: [{'attempt': 1, 'turn1': ..., 'turn2': ..., 'turn3': ...}, ...]
            'strategic_optimizations': []   # List of dicts: [{'attempt': 1, 'turn1': ..., 'turn2': ..., 'turn3': ...}, ...]
        }
    
    def record_first_attack(self, attack_results: Dict[str, Any]):
        """
        Record prompts from first attack
        
        Args:
            attack_results: Results from first attack, should contain prompt_1, prompt_2, prompt_3
        """
        if attack_results:
            self.prompts['first_attack']['turn1_prompt'] = attack_results.get('prompt_1', '')
            self.prompts['first_attack']['turn2_prompt'] = attack_results.get('prompt_2', '')
            self.prompts['first_attack']['turn3_prompt'] = attack_results.get('prompt_3', '')
    
    def record_second_attack(self, attack_results: Dict[str, Any]):
        """
        Record prompts from second attack (attack method)
        
        Args:
            attack_results: Results from second attack, should contain prompt_1, prompt_2, prompt_3
        """
        if attack_results:
            self.prompts['second_attack']['turn1_prompt'] = attack_results.get('prompt_1', '')
            self.prompts['second_attack']['turn2_prompt'] = attack_results.get('prompt_2', '')
            self.prompts['second_attack']['turn3_prompt'] = attack_results.get('prompt_3', '')
    
    def record_tactical_optimization(self, attempt: int, attack_results: Dict[str, Any]):
        """
        Record prompts from tactical optimization attack
        
        Args:
            attempt: Tactical optimization attempt number (1, 2, 3, ...)
            attack_results: Results from tactical optimization attack, should contain prompt_1, prompt_2, prompt_3
        """
        if attack_results:
            # Find existing entry or create new one
            existing = None
            for i, entry in enumerate(self.prompts['tactical_optimizations']):
                if entry.get('attempt') == attempt:
                    existing = i
                    break
            
            entry = {
                'attempt': attempt,
                'turn1_prompt': attack_results.get('prompt_1', ''),
                'turn2_prompt': attack_results.get('prompt_2', ''),
                'turn3_prompt': attack_results.get('prompt_3', '')
            }
            
            if existing is not None:
                self.prompts['tactical_optimizations'][existing] = entry
            else:
                self.prompts['tactical_optimizations'].append(entry)
    
    def record_strategic_optimization(self, attempt: int, attack_results: Dict[str, Any]):
        """
        Record prompts from strategic optimization attack
        
        Args:
            attempt: Strategic optimization attempt number (1, 2, 3, ...)
            attack_results: Results from strategic optimization attack, should contain prompt_1, prompt_2, prompt_3
        """
        if attack_results:
            # Find existing entry or create new one
            existing = None
            for i, entry in enumerate(self.prompts['strategic_optimizations']):
                if entry.get('attempt') == attempt:
                    existing = i
                    break
            
            entry = {
                'attempt': attempt,
                'turn1_prompt': attack_results.get('prompt_1', ''),
                'turn2_prompt': attack_results.get('prompt_2', ''),
                'turn3_prompt': attack_results.get('prompt_3', '')
            }
            
            if existing is not None:
                self.prompts['strategic_optimizations'][existing] = entry
            else:
                self.prompts['strategic_optimizations'].append(entry)
    
    def get_prompts_dict(self) -> Dict[str, Any]:
        """
        Get all recorded prompts as a dictionary
        
        Returns:
            Dictionary containing all recorded prompts
        """
        return self.prompts.copy()
    
    def get_csv_row(self) -> Dict[str, str]:
        """
        Get prompts formatted for CSV row
        
        Returns:
            Dictionary with keys suitable for CSV columns
        """
        row = {}
        
        # First attack prompts (convert None to empty string)
        row['first_attack_turn1_prompt'] = self.prompts['first_attack'].get('turn1_prompt') or ''
        row['first_attack_turn2_prompt'] = self.prompts['first_attack'].get('turn2_prompt') or ''
        row['first_attack_turn3_prompt'] = self.prompts['first_attack'].get('turn3_prompt') or ''
        
        # Second attack prompts (convert None to empty string)
        row['second_attack_turn1_prompt'] = self.prompts['second_attack'].get('turn1_prompt') or ''
        row['second_attack_turn2_prompt'] = self.prompts['second_attack'].get('turn2_prompt') or ''
        row['second_attack_turn3_prompt'] = self.prompts['second_attack'].get('turn3_prompt') or ''
        
        # Tactical optimization prompts (support up to 3 attempts)
        max_tactical = 3
        for i in range(1, max_tactical + 1):
            tactical_entry = next(
                (e for e in self.prompts['tactical_optimizations'] if e.get('attempt') == i),
                None
            )
            if tactical_entry:
                row[f'tactical_opt_attempt{i}_turn1_prompt'] = tactical_entry.get('turn1_prompt') or ''
                row[f'tactical_opt_attempt{i}_turn2_prompt'] = tactical_entry.get('turn2_prompt') or ''
                row[f'tactical_opt_attempt{i}_turn3_prompt'] = tactical_entry.get('turn3_prompt') or ''
            else:
                row[f'tactical_opt_attempt{i}_turn1_prompt'] = ''
                row[f'tactical_opt_attempt{i}_turn2_prompt'] = ''
                row[f'tactical_opt_attempt{i}_turn3_prompt'] = ''
        
        # Strategic optimization prompts (support up to 3 attempts)
        max_strategic = 3
        for i in range(1, max_strategic + 1):
            strategic_entry = next(
                (e for e in self.prompts['strategic_optimizations'] if e.get('attempt') == i),
                None
            )
            if strategic_entry:
                row[f'strategic_opt_attempt{i}_turn1_prompt'] = strategic_entry.get('turn1_prompt') or ''
                row[f'strategic_opt_attempt{i}_turn2_prompt'] = strategic_entry.get('turn2_prompt') or ''
                row[f'strategic_opt_attempt{i}_turn3_prompt'] = strategic_entry.get('turn3_prompt') or ''
            else:
                row[f'strategic_opt_attempt{i}_turn1_prompt'] = ''
                row[f'strategic_opt_attempt{i}_turn2_prompt'] = ''
                row[f'strategic_opt_attempt{i}_turn3_prompt'] = ''
        
        return row


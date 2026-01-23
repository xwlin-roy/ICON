"""
Template manager module
Responsible for saving and loading authoritative document templates to avoid duplicate generation
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


class TemplateManager:
    """Template manager, used to save and load authoritative document templates"""
    
    def __init__(self, template_dir: str = "templates"):
        """
        Initialize template manager
        
        Args:
            template_dir: Template save directory
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
    
    def get_template_path(self, template_type: str, is_attack: bool = False) -> Path:
        """
        Get template file path
        
        Args:
            template_type: Template type (paper, script, case_study, cti_briefing, rca_report)
            is_attack: Whether it's an attack method template (False means defense method template)
        
        Returns:
            Template file path
        """
        suffix = "_attack" if is_attack else "_defense"
        filename = f"{template_type}{suffix}.json"
        return self.template_dir / filename
    
    def save_template(self, template_type: str, template_content: str, is_attack: bool = False) -> bool:
        """
        Save template to file
        
        Args:
            template_type: Template type
            template_content: Template content (JSON string)
            is_attack: Whether it's an attack method template
        
        Returns:
            Whether save was successful
        """
        try:
            template_path = self.get_template_path(template_type, is_attack)
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            return True
        except Exception as e:
            print(f"[-] Failed to save template: {str(e)}")
            return False
    
    def load_template(self, template_type: str, is_attack: bool = False) -> Optional[str]:
        """
        Load template from file
        
        Args:
            template_type: Template type
            is_attack: Whether it's an attack method template
        
        Returns:
            Template content (JSON string), or None if it doesn't exist
        """
        template_path = self.get_template_path(template_type, is_attack)
        
        if not template_path.exists():
            return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[-] Failed to load template: {str(e)}")
            return None
    
    def template_exists(self, template_type: str, is_attack: bool = False) -> bool:
        """
        Check if template exists
        
        Args:
            template_type: Template type
            is_attack: Whether it's an attack method template
        
        Returns:
            Whether template exists
        """
        template_path = self.get_template_path(template_type, is_attack)
        return template_path.exists()
    
    def delete_template(self, template_type: str, is_attack: bool = False) -> bool:
        """
        Delete template file
        
        Args:
            template_type: Template type
            is_attack: Whether it's an attack method template
        
        Returns:
            Whether deletion was successful
        """
        template_path = self.get_template_path(template_type, is_attack)
        
        if not template_path.exists():
            return False
        
        try:
            template_path.unlink()
            return True
        except Exception as e:
            print(f"[-] Failed to delete template: {str(e)}")
            return False


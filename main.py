"""
Unified Jailbreak Attack Framework Main Entry
Integrates MoE Router, five authoritative text generators, and Multi-turn attack module
to execute complete jailbreak attack workflow
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config, LLMConfig
from src.utils.csv_reader import CSVReader
from src.utils.logger import init_logger
from src.utils.sample_logger import SampleLogger
from src.utils.metrics import MetricsCalculator, print_m2s_summary
from src.utils.cost_tracker import CostTracker
from src.utils.prompt_recorder import PromptRecorder
from src.router import MoERouter
from src.forgery import (
    PaperGenerator,
    ScriptGenerator,
    CaseStudyGenerator,
    CTIBriefingGenerator,
    RCAReportGenerator
)
from src.forgery.template_manager import TemplateManager
from src.attack.multi_turn_attacker import MultiTurnAttacker
from src.judge.judge_llm import JudgeLLM
from src.reflector import Reflector


class UnifiedJailbreakFramework:
    """Unified jailbreak attack framework main class, integrating all modules"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the framework
        
        Args:
            config: Configuration object, if None then use default configuration
        """
        self.config = config or Config()
        self.config.validate()
        
        # Get target LLM name for log directory
        target_llm_name = self.config.target_llm.model_name
        
        # Initialize logger with target_llm_name to create subdirectory
        # Optimization: Disable file logging by default, only keep console error output to reduce I/O overhead
        # To enable detailed logging, set "logging.log_to_file": true in config.json
        log_to_file = getattr(self.config, 'log_to_file', False)
        if hasattr(self.config, 'config_data') and 'logging' in self.config.config_data:
            log_to_file = self.config.config_data.get('logging', {}).get('log_to_file', False)
        
        init_logger(
            log_dir=self.config.log_dir,
            log_level=self.config.log_level,
            log_to_file=log_to_file,  # Disable file logging by default
            log_to_console=self.config.log_to_console,
            log_sample_rate=0.0 if not log_to_file else getattr(self.config, 'log_sample_rate', 0.0),  # Set sampling rate to 0 when file logging is disabled
            log_first_n=0 if not log_to_file else getattr(self.config, 'log_first_n', 0),  # Do not log when file logging is disabled
            target_llm_name=target_llm_name
        )
        
        # Initialize MoE Router
        self.router = MoERouter(self.config.router_llm)
        
        # Initialize all authoritative text generators
        self.paper_generator = PaperGenerator(self.config.paper_generator_llm)
        self.script_generator = ScriptGenerator(self.config.script_generator_llm)
        self.case_study_generator = CaseStudyGenerator(self.config.case_study_generator_llm)
        self.cti_briefing_generator = CTIBriefingGenerator(self.config.cti_briefing_generator_llm)
        self.rca_report_generator = RCAReportGenerator(self.config.rca_report_generator_llm)
        
        # Initialize attacker and Judge
        # Target Model (the model being attacked) uses temperature=0.0 to ensure greedy decoding and consistent attack results
        target_llm_config = LLMConfig(
            model_name=self.config.target_llm.model_name,
            provider=self.config.target_llm.provider,
            api_key=self.config.target_llm.api_key,
            api_base=self.config.target_llm.api_base,
            temperature=0.0,  # Target Model uses temperature=0.0 to ensure greedy decoding and consistent attack results
            max_tokens=self.config.target_llm.max_tokens
        )
        
        # Attacker Model (used by Prompt Optimizer) uses temperature=0.7 to ensure attack diversity
        attacker_llm_config = LLMConfig(
            model_name=self.config.reflector_llm.model_name,
            provider=self.config.reflector_llm.provider,
            api_key=self.config.reflector_llm.api_key,
            api_base=self.config.reflector_llm.api_base,
            temperature=0.7,  # Attacker Model uses temperature=0.7 to ensure attack diversity
            max_tokens=self.config.reflector_llm.max_tokens
        )
        
        self.attacker = MultiTurnAttacker(
            target_llm_config,  # Target Model config (temperature=0.0)
            optimizer_llm_config=attacker_llm_config  # Attacker Model config (temperature=0.7)
        )
        self.judge = JudgeLLM(self.config.judge_llm)
        
        # Initialize Reflector
        self.reflector = Reflector(self.config.reflector_llm)
        
        # Initialize sample logger (also use target_llm_name for subdirectory)
        # Note: Disable detailed logging by default to reduce I/O overhead
        # To enable, set "enable_sample_logging": true in config.json
        enable_sample_logging = getattr(self.config, 'enable_sample_logging', False) or \
                                (hasattr(self.config, 'config_data') and 
                                 self.config.config_data.get('logging', {}).get('enable_sample_logging', False))
        
        if enable_sample_logging:
            sample_log_dir = str(Path(self.config.log_dir) / target_llm_name.replace('/', '_').replace('\\', '_').replace(':', '_') / "samples_multiturn")
            self.sample_logger = SampleLogger(log_dir=sample_log_dir)
        else:
            self.sample_logger = None  # Disable detailed logging
        
        # Initialize template manager
        template_dir = str(Path(self.config.output_dir).parent / "templates")
        self.template_manager = TemplateManager(template_dir=template_dir)
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _get_generator_by_pattern(self, pattern: str):
        """
        Get the corresponding generator by pattern
        
        Args:
            pattern: Context pattern name
        
        Returns:
            Corresponding generator instance
        """
        pattern_to_generator = {
            "Scientific Research": self.paper_generator,
            "Personal Narrative": self.case_study_generator,
            "Fictional Scenario": self.script_generator,
            "Information Retrieval": self.cti_briefing_generator,
            "Problem Solving": self.rca_report_generator
        }
        return pattern_to_generator.get(pattern)
    
    def _get_generator_config_by_pattern(self, pattern: str):
        """
        Get the corresponding generator configuration by pattern
        
        Args:
            pattern: Context pattern name
        
        Returns:
            Corresponding generator LLM configuration
        """
        pattern_to_config = {
            "Scientific Research": self.config.paper_generator_llm,
            "Personal Narrative": self.config.case_study_generator_llm,
            "Fictional Scenario": self.config.script_generator_llm,
            "Information Retrieval": self.config.cti_briefing_generator_llm,
            "Problem Solving": self.config.rca_report_generator_llm
        }
        return pattern_to_config.get(pattern)
    
    def _feedback_optimization_loop(
        self,
        harmful_query: str,
        pattern: str,
        document_type: str,
        generator,
        generator_config,
        forged_document: str,
        first_attack_results: Optional[dict] = None,
        routing_result: dict = None,
        sample_index: Optional[int] = None,
        cost_tracker: Optional[CostTracker] = None,
        prompt_recorder: Optional[PromptRecorder] = None
    ) -> tuple:
        """
        Dual-loop feedback optimization mechanism
        
        Args:
            harmful_query: Original harmful query
            pattern: Currently selected pattern
            document_type: Current document type
            generator: Generator instance
            generator_config: Generator configuration
            forged_document: Generated defense method document
            routing_result: Router routing result
            sample_index: Sample index
        
        Returns:
            (jailbreak_response, judge_result) tuple
        """
        max_tactical_retries = self.config.max_tactical_retries
        max_strategic_retries = self.config.max_strategic_retries
        
        # Initialize optimization statistics
        optimization_stats = {
            'second_attack_success': False,  # Track if second attack (attack method) succeeded
            'tactical_success': False,
            'strategic_success': False,
            'tactical_attempts': 0,
            'strategic_attempts': 0
        }
        
        # Step 1: Try second attack (attack method)
        print("[*] Phase 2.5.1: Attempting second attack (attack method)...")
        attack_document, attack_results_2 = self._try_attack_method(
            harmful_query, pattern, document_type, generator, generator_config, sample_index, cost_tracker=cost_tracker
        )
        
        # Record second attack prompts
        if prompt_recorder is not None and attack_results_2:
            prompt_recorder.record_second_attack(attack_results_2)
        
        if attack_document and attack_results_2:
            # Extract response and judge
            extracted_response_2 = attack_results_2.get('extracted_jailbreak_response', {})
            jailbreak_response_2 = extracted_response_2.get('hypothetical_llm_response', '')
            if not jailbreak_response_2:
                jailbreak_response_2 = attack_results_2.get('final_response', '')
            
            if jailbreak_response_2 and jailbreak_response_2.strip():
                print("[*] Judging second attack result...")
                return_details = sample_index is not None
                judge_result_2 = self._judge_with_retry(
                    harmful_query, jailbreak_response_2, return_details, sample_index, max_retries=2, cost_tracker=cost_tracker
                )
                
                if judge_result_2:
                    if judge_result_2.get('is_harmful', False):
                        print("[+] Second attack judge succeeded!")
                        optimization_stats['second_attack_success'] = True  # Mark second attack as successful
                        return jailbreak_response_2, judge_result_2, optimization_stats
                    else:
                        print("[!] Second attack judge failed, entering tactical optimization loop...")
                else:
                    print("[!] Second attack judge failed after retries, entering tactical optimization loop...")
        
        # Step 2: Tactical optimization loop
        print(f"[*] Phase 2.5.2: Entering tactical optimization loop (max {max_tactical_retries} attempts)...")
        current_document = attack_document if attack_document else forged_document
        current_attack_results = attack_results_2 if attack_results_2 else first_attack_results
        
        # Check if we have valid attack results for tactical optimization
        can_perform_tactical = False
        if current_attack_results:
            attack_prompt_3 = current_attack_results.get('prompt_3', '')
            attack_response_3 = current_attack_results.get('response_3', current_attack_results.get('turn_3', ''))
            if attack_prompt_3 and attack_response_3:
                can_perform_tactical = True
        
        if not can_perform_tactical:
            print("[!] Cannot get third turn prompt and response, skipping tactical optimization loop")
            # Mark that tactical optimization was attempted but skipped due to missing data
            optimization_stats['tactical_attempts'] = 0  # Set to 0 to indicate it was skipped, not attempted
        else:
            for tactical_attempt in range(1, max_tactical_retries + 1):
                optimization_stats['tactical_attempts'] = tactical_attempt
                print(f"[*] Tactical optimization attempt {tactical_attempt}/{max_tactical_retries}...")
                
                # Get third turn prompt and response (we already checked they exist)
                attack_prompt_3 = current_attack_results.get('prompt_3', '')
                attack_response_3 = current_attack_results.get('response_3', current_attack_results.get('turn_3', ''))
                
                # Call Reflector for tactical reflection
                try:
                    return_details = sample_index is not None
                    tactical_reflection = self.reflector.reflect_tactical(
                        harmful_query, attack_prompt_3, attack_response_3, return_details
                    )
                    
                    # Record Reflector cost (tactical)
                    if cost_tracker is not None:
                        metadata = self.reflector.llm_client.get_last_metadata()
                        cost_tracker.record_api_call('reflector', metadata.get('token_usage'))
                    
                    if tactical_reflection.get('flag') == 'TACTICAL_ERROR':
                        tactical_feedback = tactical_reflection.get('tactical_feedback', '')
                        error_type = tactical_reflection.get('error_type', 'unknown')
                        
                        print(f"[+] Tactical reflection completed: {error_type}")
                        print(f"[*] Tactical feedback: {tactical_feedback[:100]}...")
                        
                        # Log Reflector interaction (only if sample logging enabled)
                        if return_details and sample_index is not None and self.sample_logger is not None:
                            metadata = self.reflector.llm_client.get_last_metadata()
                            self.sample_logger.log_interaction(
                                stage=f'tactical_reflection_attempt_{tactical_attempt}',
                                model_name=self.config.reflector_llm.model_name,
                                prompt=tactical_reflection.get('prompt', ''),
                                response=tactical_reflection.get('response', ''),
                                temperature=self.config.reflector_llm.temperature,
                                max_tokens=self.config.reflector_llm.max_tokens,
                                response_time=metadata.get('response_time'),
                                token_usage=metadata.get('token_usage')
                            )
                        
                        # Re-attack with optimized prompt (only execute third turn, keep first two turns unchanged)
                        # Note: Need to maintain conversation history of first two turns
                        self.attacker.reset()  # Reset conversation history
                        
                        # Re-execute complete three-turn attack, but third turn uses optimized prompt
                        return_details = sample_index is not None
                        optimized_attack_results = self.attacker.attack_with_attack_scenario(
                            current_document,
                            document_type,
                            return_details=return_details,
                            tactical_feedback=tactical_feedback,
                            error_type=error_type
                        )
                        
                        # Record tactical optimization prompts
                        if prompt_recorder is not None and optimized_attack_results:
                            prompt_recorder.record_tactical_optimization(tactical_attempt, optimized_attack_results)
                        
                        # Record Target LLM cost (tactical optimization attack)
                        if cost_tracker is not None:
                            turn_metadata_list = optimized_attack_results.get('turn_metadata', [])
                            for turn_metadata in turn_metadata_list:
                                cost_tracker.record_api_call('target_llm', turn_metadata.get('token_usage'))
                        
                        # Log optimized attack (only if sample logging enabled)
                        if return_details and self.sample_logger is not None:
                            turn_metadata_list = optimized_attack_results.get('turn_metadata', [])
                            for turn_num in range(1, 4):
                                prompt_key = f'prompt_{turn_num}'
                                response_key = f'response_{turn_num}'
                                if prompt_key in optimized_attack_results:
                                    turn_metadata = turn_metadata_list[turn_num - 1] if turn_num <= len(turn_metadata_list) else {}
                                    self.sample_logger.log_interaction(
                                        stage=f'tactical_optimized_attack_attempt_{tactical_attempt}_turn_{turn_num}',
                                        model_name=self.config.target_llm.model_name,
                                        prompt=optimized_attack_results[prompt_key],
                                        response=optimized_attack_results.get(response_key, optimized_attack_results.get(f'turn_{turn_num}', '')),
                                        temperature=self.config.target_llm.temperature,
                                        max_tokens=self.config.target_llm.max_tokens,
                                        response_time=turn_metadata.get('response_time'),
                                        token_usage=turn_metadata.get('token_usage')
                                    )
                        
                        # Extract response
                        extracted_optimized = optimized_attack_results.get('extracted_jailbreak_response', {})
                        optimized_response = extracted_optimized.get('hypothetical_llm_response', '')
                        if not optimized_response:
                            optimized_response = optimized_attack_results.get('final_response', '')
                        
                        if optimized_response and optimized_response.strip():
                            # Must perform judge evaluation
                            print(f"[*] Judging tactical optimization attempt {tactical_attempt} result...")
                            return_details = sample_index is not None
                            judge_result_optimized = self._judge_with_retry(
                                harmful_query, optimized_response, return_details, sample_index, max_retries=2, cost_tracker=cost_tracker
                            )
                            
                            if judge_result_optimized:
                                if judge_result_optimized.get('is_harmful', False):
                                    print(f"[+] Tactical optimization attempt {tactical_attempt} succeeded!")
                                    optimization_stats['tactical_success'] = True
                                    return optimized_response, judge_result_optimized, optimization_stats
                                else:
                                    # Update current results, continue next optimization
                                    current_attack_results = optimized_attack_results
                                    current_document = current_document  # Document unchanged
                                    print(f"[!] Tactical optimization attempt {tactical_attempt} judge failed, continuing optimization...")
                            else:
                                print(f"[!] Tactical optimization attempt {tactical_attempt} judge failed after retries, continuing optimization...")
                                current_attack_results = optimized_attack_results
                        else:
                            print(f"[!] Tactical optimization attempt {tactical_attempt} failed to extract response, continuing optimization...")
                            # Even without response, update current_attack_results for next use
                            current_attack_results = optimized_attack_results
                    else:
                        print("[!] Reflector did not identify tactical error, skipping tactical optimization")
                        break
                        
                except Exception as e:
                    print(f"[-] Tactical reflection failed: {str(e)}")
                    # Only log errors (even if logging is disabled, errors should be logged)
                    if sample_index is not None and self.sample_logger is not None:
                        self.sample_logger.log_interaction(
                            stage=f'tactical_reflection_attempt_{tactical_attempt}',
                            model_name=self.config.reflector_llm.model_name,
                            prompt='',
                            response='',
                            error=str(e)
                        )
                    break
        
        # Step 3: Strategic optimization loop
        print(f"[*] Phase 2.5.3: Entering strategic optimization loop (max {max_strategic_retries} attempts)...")
        
        for strategic_attempt in range(1, max_strategic_retries + 1):
            optimization_stats['strategic_attempts'] = strategic_attempt
            print(f"[*] Strategic optimization attempt {strategic_attempt}/{max_strategic_retries}...")
            
            # Get current attack's third turn prompt and response
            if current_attack_results:
                final_prompt_3 = current_attack_results.get('prompt_3', '')
                final_response_3 = current_attack_results.get('response_3', current_attack_results.get('turn_3', ''))
            else:
                final_prompt_3 = ''
                final_response_3 = ''
            
            if not final_prompt_3 or not final_response_3:
                print("[!] Cannot get attack prompt and response, skipping strategic optimization")
                break
            
            # Call Reflector for strategic reflection
            try:
                return_details = sample_index is not None
                strategic_reflection = self.reflector.reflect_strategic(
                    harmful_query=harmful_query,
                    intent_category=routing_result.get('intent_category', 'Unknown'),
                    selected_pattern=pattern,
                    pattern_reasoning=routing_result.get('reasoning', ''),
                    attack_prompt=final_prompt_3,
                    model_response=final_response_3,
                    return_details=return_details
                )
                
                # Record Reflector cost (strategic)
                if cost_tracker is not None:
                    metadata = self.reflector.llm_client.get_last_metadata()
                    cost_tracker.record_api_call('reflector', metadata.get('token_usage'))
                
                if strategic_reflection.get('flag') == 'STRATEGIC_ERROR':
                    strategic_feedback = strategic_reflection.get('strategic_feedback', '')
                    
                    print(f"[+] Strategic reflection completed")
                    print(f"[*] Strategic feedback: {strategic_feedback[:150]}...")
                    
                    # Log Reflector interaction (only if sample logging enabled)
                    if return_details and sample_index is not None and self.sample_logger is not None:
                        metadata = self.reflector.llm_client.get_last_metadata()
                        self.sample_logger.log_interaction(
                            stage=f'strategic_reflection_attempt_{strategic_attempt}',
                            model_name=self.config.reflector_llm.model_name,
                            prompt=strategic_reflection.get('prompt', ''),
                            response=strategic_reflection.get('response', ''),
                            temperature=self.config.reflector_llm.temperature,
                            max_tokens=self.config.reflector_llm.max_tokens,
                            response_time=metadata.get('response_time'),
                            token_usage=metadata.get('token_usage')
                        )
                    
                    # Let Router re-select pattern based on strategic_feedback
                    print("[*] Router re-selecting pattern based on strategic_feedback...")
                    return_details = sample_index is not None
                    new_routing_result = self.router.route(
                        harmful_query,
                        return_details=return_details,
                        strategic_feedback=strategic_feedback
                    )
                    
                    # Record Router cost (strategic re-routing)
                    if cost_tracker is not None:
                        metadata = self.router.llm_client.get_last_metadata()
                        cost_tracker.record_api_call('router', metadata.get('token_usage'))
                    
                    new_pattern = new_routing_result.get('pattern', pattern)
                    new_document_type = new_routing_result.get('document_type', document_type)
                    
                    print(f"[+] Router re-selected: {new_pattern} -> {new_document_type}")
                    
                    # If pattern changed, need to regenerate document
                    if new_pattern != pattern:
                        # Get new generator
                        new_generator = self._get_generator_by_pattern(new_pattern)
                        new_generator_config = self._get_generator_config_by_pattern(new_pattern)
                        
                        if new_generator:
                            # Regenerate document
                            print(f"[*] Regenerating {document_type} (new pattern: {new_pattern})...")
                            return_details = sample_index is not None
                            new_doc_result = self._generate_with_template(
                                new_generator, new_pattern, harmful_query,
                                is_attack=False, return_details=return_details
                            )
                            
                            # Record Generator cost (strategic regeneration)
                            if cost_tracker is not None and not new_doc_result.get('from_template', False):
                                metadata = new_generator.llm_client.get_last_metadata()
                                cost_tracker.record_api_call('generator', metadata.get('token_usage'))
                            
                            if return_details:
                                if new_pattern == "Scientific Research":
                                    new_forged_doc = new_doc_result.get('paper', '')
                                elif new_pattern == "Personal Narrative":
                                    new_forged_doc = new_doc_result.get('case_study', '')
                                elif new_pattern == "Fictional Scenario":
                                    new_forged_doc = new_doc_result.get('script', '')
                                elif new_pattern == "Information Retrieval":
                                    new_forged_doc = new_doc_result.get('briefing', '')
                                elif new_pattern == "Problem Solving":
                                    new_forged_doc = new_doc_result.get('report', '')
                                else:
                                    new_forged_doc = new_doc_result.get(list(new_doc_result.keys())[0], '')
                            else:
                                new_forged_doc = new_doc_result
                            
                            # Reset attacker and execute new attack
                            self.attacker.reset()
                            return_details = sample_index is not None
                            new_attack_results = self.attacker.attack(
                                new_forged_doc, new_document_type, return_details=return_details
                            )
                            
                            # Record strategic optimization prompts
                            if prompt_recorder is not None and new_attack_results:
                                prompt_recorder.record_strategic_optimization(strategic_attempt, new_attack_results)
                            
                            # Record Target LLM cost (strategic optimization attack)
                            if cost_tracker is not None:
                                turn_metadata_list = new_attack_results.get('turn_metadata', [])
                                for turn_metadata in turn_metadata_list:
                                    cost_tracker.record_api_call('target_llm', turn_metadata.get('token_usage'))
                            
                            # Extract response
                            extracted_new = new_attack_results.get('extracted_jailbreak_response', {})
                            new_response = extracted_new.get('hypothetical_llm_response', '')
                            if not new_response:
                                new_response = new_attack_results.get('final_response', '')
                            
                            if new_response and new_response.strip():
                                # Must perform judge evaluation
                                print(f"[*] Judging strategic optimization attempt {strategic_attempt} result...")
                                return_details = sample_index is not None
                                new_judge_result = self._judge_with_retry(
                                    harmful_query, new_response, return_details, sample_index, max_retries=2, cost_tracker=cost_tracker
                                )
                                
                                if new_judge_result:
                                    if new_judge_result.get('is_harmful', False):
                                        print(f"[+] Strategic optimization attempt {strategic_attempt} succeeded!")
                                        optimization_stats['strategic_success'] = True
                                        # Update routing_result
                                        routing_result.update(new_routing_result)
                                        return new_response, new_judge_result, optimization_stats
                                    else:
                                        print(f"[!] Strategic optimization attempt {strategic_attempt} judge failed")
                                        # Update current results, continue next optimization (if any)
                                        current_attack_results = new_attack_results
                                        current_document = new_forged_doc
                                        pattern = new_pattern
                                        document_type = new_document_type
                                        routing_result.update(new_routing_result)
                                else:
                                    print(f"[!] Strategic optimization attempt {strategic_attempt} judge failed after retries")
                                    current_attack_results = new_attack_results
                                    current_document = new_forged_doc
                                    pattern = new_pattern
                                    document_type = new_document_type
                                    routing_result.update(new_routing_result)
                            else:
                                print(f"[!] Strategic optimization attempt {strategic_attempt} failed to extract response")
                                # Even without response, update current_attack_results for next use
                                current_attack_results = new_attack_results
                                current_document = new_forged_doc
                                pattern = new_pattern
                                document_type = new_document_type
                                routing_result.update(new_routing_result)
                    else:
                        print("[!] Router selected the same pattern, skipping regeneration")
                        break
                else:
                    print("[!] Reflector did not identify strategic error, skipping strategic optimization")
                    break
                    
            except Exception as e:
                print(f"[-] Strategic reflection failed: {str(e)}")
                # 只记录错误（即使日志被禁用，错误也应该记录）
                if sample_index is not None and self.sample_logger is not None:
                    self.sample_logger.log_interaction(
                        stage=f'strategic_reflection_attempt_{strategic_attempt}',
                        model_name=self.config.reflector_llm.model_name,
                        prompt='',
                        response='',
                        error=str(e)
                    )
                break
        
        # All optimizations failed, return failure result
        # Note: If we have a final response but no judge result, we should still try to judge it
        print("[!] All feedback optimization attempts failed, abandoning this attack")
        
        # Try to judge the final response if we have one
        final_response = ''
        final_judge_result = None
        
        # Get the last response we tried
        if current_attack_results:
            extracted_final = current_attack_results.get('extracted_jailbreak_response', {})
            final_response = extracted_final.get('hypothetical_llm_response', '')
            if not final_response:
                final_response = current_attack_results.get('final_response', '')
        
        # If we have a response, try to judge it (even if optimization failed)
        if final_response and final_response.strip():
            print("[*] Attempting to judge final response despite optimization failure...")
            try:
                return_details = sample_index is not None
                final_judge_result = self._judge_with_retry(
                    harmful_query, final_response, return_details, sample_index, max_retries=2, cost_tracker=cost_tracker
                )
                if final_judge_result:
                    print(f"[+] Final judge completed: Score = {final_judge_result.get('score', 0.0):.3f}")
            except Exception as e:
                print(f"[!] Final judge failed after retries: {str(e)}")
                final_judge_result = None
        
        # If we have a judge result, return it; otherwise mark as needs retry
        if final_judge_result:
            return final_response, final_judge_result, optimization_stats
        else:
            # No valid judge result - mark as needs retry
            # Note: This should rarely happen now since _judge_with_retry always returns a dict
            return final_response, {
                'score': 0.0,  # Ensure score field is present for proper filtering
                'is_harmful': False,
                'is_jailbroken': False,
                'refused': True,
                'convincing': 1,
                'specific': 1,
                'needs_retry': True,  # Mark that this sample needs to be retested
                'explanation': "Failed to jailbreak after tactical and strategic optimization. Judge failed - needs retry.",
                'raw_response': ''
            }, optimization_stats
    
    def _try_attack_method(
        self,
        harmful_query: str,
        pattern: str,
        document_type: str,
        generator,
        generator_config,
        sample_index: Optional[int] = None,
        cost_tracker: Optional[CostTracker] = None
    ) -> tuple:
        """
        Try second attack (attack method)
        
        Args:
            harmful_query: Harmful query
            pattern: Pattern name
            document_type: Document type
            generator: Generator instance
            generator_config: Generator configuration
            sample_index: Sample index
        
        Returns:
            (attack_document, attack_results) tuple
        """
        try:
            print(f"[*] Regenerating attack method {document_type}...")
            return_details = sample_index is not None
            
            # Use template manager to generate attack method document
            attack_doc_result = self._generate_with_template(
                generator, pattern, harmful_query, is_attack=True, return_details=return_details
            )
            
            # Record Generator cost (attack method generation)
            if cost_tracker is not None and not attack_doc_result.get('from_template', False):
                metadata = generator.llm_client.get_last_metadata()
                cost_tracker.record_api_call('generator', metadata.get('token_usage'))
            
            # Get document based on pattern
            if return_details:
                if pattern == "Scientific Research":
                    attack_document = attack_doc_result.get('paper', '')
                elif pattern == "Personal Narrative":
                    attack_document = attack_doc_result.get('case_study', '')
                elif pattern == "Fictional Scenario":
                    attack_document = attack_doc_result.get('script', '')
                elif pattern == "Information Retrieval":
                    attack_document = attack_doc_result.get('briefing', '')
                elif pattern == "Problem Solving":
                    attack_document = attack_doc_result.get('report', '')
                else:
                    attack_document = attack_doc_result.get(list(attack_doc_result.keys())[0], '')
            else:
                attack_document = attack_doc_result
            
            print(f"[+] Attack method {document_type} generation completed")
            
            # Reset attacker conversation history
            self.attacker.reset()
            
            # Execute second attack
            return_details = sample_index is not None
            attack_results_2 = self.attacker.attack_with_attack_scenario(
                attack_document,
                document_type,
                return_details=return_details
            )
            
            # Record Target LLM cost (second attack)
            if cost_tracker is not None:
                turn_metadata_list = attack_results_2.get('turn_metadata', [])
                for turn_metadata in turn_metadata_list:
                    cost_tracker.record_api_call('target_llm', turn_metadata.get('token_usage'))
            
            # Log attack interactions (only if sample logging enabled)
            if return_details and self.sample_logger is not None:
                turn_metadata_list = attack_results_2.get('turn_metadata', [])
                for turn_num in range(1, 4):
                    prompt_key = f'prompt_{turn_num}'
                    response_key = f'response_{turn_num}'
                    if prompt_key in attack_results_2:
                        turn_metadata = turn_metadata_list[turn_num - 1] if turn_num <= len(turn_metadata_list) else {}
                        self.sample_logger.log_interaction(
                            stage=f'multi_turn_attack_attempt_2_turn_{turn_num}',
                            model_name=self.config.target_llm.model_name,
                            prompt=attack_results_2[prompt_key],
                            response=attack_results_2.get(response_key, attack_results_2.get(f'turn_{turn_num}', '')),
                            temperature=self.config.target_llm.temperature,
                            max_tokens=self.config.target_llm.max_tokens,
                            response_time=turn_metadata.get('response_time'),
                            token_usage=turn_metadata.get('token_usage')
                        )
            
            return attack_document, attack_results_2
            
        except Exception as e:
            print(f"[-] Second attack failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _judge_with_retry(
        self,
        harmful_query: str,
        jailbreak_response: str,
        return_details: bool = False,
        sample_index: Optional[int] = None,
        max_retries: int = 2,
        cost_tracker: Optional[CostTracker] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Judge jailbreak response with retry mechanism
        
        Args:
            harmful_query: Original harmful query
            jailbreak_response: Jailbreak response to judge
            return_details: Whether to return detailed information
            sample_index: Sample index for logging (optional)
            max_retries: Maximum number of retry attempts (default: 2)
        
        Returns:
            Judge result dictionary with 'score' field (always present), or None if all retries failed
            Note: Even if Judge LLM call fails, judge() method will return a valid result dict
            with score=0.0. This method only returns None if there's a critical error.
        """
        import time
        
        # Check for empty response - should not call Judge on empty response
        if not jailbreak_response or not jailbreak_response.strip():
            # Return default failed result for empty response
            result = {
                "score": 0.0,
                "refused": True,
                "convincing": 1,
                "specific": 1,
                "is_jailbroken": False,
                "is_perfect": False,
                "is_harmful": False,
                "reasoning": "Empty jailbreak response - automatically judged as failure",
                "raw_response": "",
                "explanation": "Empty jailbreak response - automatically judged as failure (score=0.0)"
            }
            if return_details:
                result['prompt'] = ""
                result['response'] = ""
            return result
        
        # Retry loop for network/model errors
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Call judge (judge() method now handles exceptions internally and always returns valid dict)
                judge_result = self.judge.judge(
                    harmful_behavior=harmful_query,
                    generation=jailbreak_response,
                    return_details=return_details
                )
                
                # Ensure result has score field (should always be present now)
                if 'score' not in judge_result:
                    judge_result['score'] = 0.0
                
                # Record Judge cost
                if cost_tracker is not None:
                    metadata = self.judge.llm_client.get_last_metadata()
                    cost_tracker.record_api_call('judge', metadata.get('token_usage'))
                
                # Log interaction if needed
                if return_details and sample_index is not None and self.sample_logger is not None:
                    if 'prompt' in judge_result:
                        metadata = self.judge.llm_client.get_last_metadata()
                        self.sample_logger.log_interaction(
                            stage=f'judge_attempt_{attempt + 1}',
                            model_name=self.config.judge_llm.model_name,
                            prompt=judge_result['prompt'],
                            response=judge_result.get('response', judge_result.get('raw_response', '')),
                            temperature=self.config.judge_llm.temperature,
                            max_tokens=self.config.judge_llm.max_tokens,
                            response_time=metadata.get('response_time'),
                            token_usage=metadata.get('token_usage')
                        )
                
                # Check if this was a failed Judge LLM call (indicated by error in explanation)
                if 'failed' in judge_result.get('explanation', '').lower():
                    # Judge LLM call failed but returned a valid result
                    # Check if we should retry (network errors, not empty response)
                    if attempt < max_retries and 'network' in judge_result.get('explanation', '').lower() or 'connection' in judge_result.get('explanation', '').lower():
                        last_exception = Exception(judge_result.get('explanation', 'Judge call failed'))
                        wait_time = min(1.0 * (2 ** attempt), 10.0)
                        print(f"[!] Judge attempt {attempt + 1} failed: {judge_result.get('explanation')}. Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        continue
                    # If not retryable or retries exhausted, return the result (score=0.0)
                    return judge_result
                
                # Success - return result
                return judge_result
                
            except Exception as e:
                # Unexpected exception (should not happen now that judge() handles exceptions)
                last_exception = e
                
                # Log error if needed
                if sample_index is not None and self.sample_logger is not None:
                    self.sample_logger.log_interaction(
                        stage=f'judge_attempt_{attempt + 1}',
                        model_name=self.config.judge_llm.model_name,
                        prompt='',
                        response='',
                        error=str(e)
                    )
                
                # Check if we have retries left
                if attempt < max_retries:
                    # Wait before retrying (exponential backoff)
                    wait_time = min(1.0 * (2 ** attempt), 10.0)  # Max 10 seconds
                    print(f"[!] Judge attempt {attempt + 1} failed with exception: {str(e)}. Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # All retries exhausted - return result indicating failure with needs_retry flag
                    print(f"[-] Judge failed after {max_retries + 1} attempts: {str(e)}")
                    # Return a result dict with score=0.0 and needs_retry flag
                    result = {
                        "score": 0.0,
                        "refused": True,
                        "convincing": 1,
                        "specific": 1,
                        "is_jailbroken": False,
                        "is_perfect": False,
                        "is_harmful": False,
                        "needs_retry": True,  # Mark for retry due to network/model failure
                        "reasoning": f"Judge failed after {max_retries + 1} attempts: {str(e)}",
                        "raw_response": "",
                        "explanation": f"Judge failed after retries - likely network/model issue: {str(e)}"
                    }
                    if return_details:
                        result['prompt'] = ""
                        result['response'] = ""
                    return result
        
        # Should not reach here, but return failure result if we do
        if last_exception:
            result = {
                "score": 0.0,
                "refused": True,
                "convincing": 1,
                "specific": 1,
                "is_jailbroken": False,
                "is_perfect": False,
                "is_harmful": False,
                "needs_retry": True,
                "reasoning": f"Judge failed: {str(last_exception)}",
                "raw_response": "",
                "explanation": f"Judge failed: {str(last_exception)}"
            }
            if return_details:
                result['prompt'] = ""
                result['response'] = ""
            return result
        
        return None
    
    def _get_template_type_by_pattern(self, pattern: str) -> str:
        """
        Get the corresponding template type name by pattern
        
        Args:
            pattern: Context pattern name
        
        Returns:
            Template type name
        """
        pattern_to_type = {
            "Scientific Research": "paper",
            "Personal Narrative": "case_study",
            "Fictional Scenario": "script",
            "Information Retrieval": "cti_briefing",
            "Problem Solving": "rca_report"
        }
        return pattern_to_type.get(pattern, "unknown")
    
    def _generate_with_template(
        self,
        generator,
        pattern: str,
        harmful_query: str,
        is_attack: bool,
        return_details: bool = False
    ) -> dict:
        """
        Generate document using template (reuse if template exists, otherwise generate new template)
        
        Args:
            generator: Generator instance
            pattern: Conversation pattern name
            harmful_query: Harmful query text
            is_attack: Whether it's an attack method document
            return_details: Whether to return detailed information
        
        Returns:
            Generated document result (same format as generator.generate)
        """
        template_type = self._get_template_type_by_pattern(pattern)
        
        # Check if template exists
        if self.template_manager.template_exists(template_type, is_attack):
            # Use template
            print(f"[*] Using saved template ({template_type}{'_attack' if is_attack else '_defense'})...")
            template_content = self.template_manager.load_template(template_type, is_attack)
            
            if template_content:
                # Call corresponding insertion method based on pattern and is_attack
                if pattern == "Scientific Research":
                    if is_attack:
                        forged_document = generator._insert_attack_scenario_section(template_content, harmful_query)
                    else:
                        forged_document = generator._insert_scenario_section(template_content, harmful_query)
                elif pattern == "Personal Narrative":
                    if is_attack:
                        forged_document = generator._insert_attack_scenario_section(template_content, harmful_query)
                    else:
                        forged_document = generator._insert_scenario_section(template_content, harmful_query)
                elif pattern == "Fictional Scenario":
                    if is_attack:
                        forged_document = generator._insert_attack_application_section(template_content, harmful_query)
                    else:
                        forged_document = generator._insert_application_section(template_content, harmful_query)
                elif pattern == "Information Retrieval":
                    if is_attack:
                        forged_document = generator._insert_attack_implementation_section(template_content, harmful_query)
                    else:
                        forged_document = generator._insert_implementation_section(template_content, harmful_query)
                elif pattern == "Problem Solving":
                    if is_attack:
                        forged_document = generator._insert_attack_solution_section(template_content, harmful_query)
                    else:
                        forged_document = generator._insert_solution_section(template_content, harmful_query)
                else:
                    # If cannot determine insertion method, fallback to directly calling generator
                    raise ValueError(f"Cannot determine insertion method for pattern '{pattern}'")
                
                # Return result
                if return_details:
                    # Determine return key
                    if pattern == "Scientific Research":
                        key = 'paper'
                    elif pattern == "Personal Narrative":
                        key = 'case_study'
                    elif pattern == "Fictional Scenario":
                        key = 'script'
                    elif pattern == "Information Retrieval":
                        key = 'briefing'
                    elif pattern == "Problem Solving":
                        key = 'report'
                    else:
                        key = 'document'
                    
                    return {
                        key: forged_document,
                        'prompt': f"[Template] Using saved template for {template_type}",
                        'response': template_content,
                        'from_template': True
                    }
                else:
                    return forged_document
        
        # Template does not exist, generate new template
        print(f"[*] Template does not exist, generating new template ({template_type}{'_attack' if is_attack else '_defense'})...")
        
        if is_attack:
            if pattern == "Scientific Research":
                result = generator.generate_attack_paper(harmful_query, return_details=True)
            elif pattern == "Personal Narrative":
                result = generator.generate_attack_case_study(harmful_query, return_details=True)
            elif pattern == "Fictional Scenario":
                result = generator.generate_attack_script(harmful_query, return_details=True)
            elif pattern == "Information Retrieval":
                result = generator.generate_attack_briefing(harmful_query, return_details=True)
            elif pattern == "Problem Solving":
                result = generator.generate_attack_report(harmful_query, return_details=True)
            else:
                raise ValueError(f"Unknown pattern: {pattern}")
        else:
            result = generator.generate(harmful_query, return_details=True)
        
        # Save template (save response, i.e., base document without malicious query)
        template_content = result.get('response', '')
        if template_content:
            self.template_manager.save_template(template_type, template_content, is_attack)
            print(f"[+] Template saved: {template_type}{'_attack' if is_attack else '_defense'}")
        
        # Return result
        if return_details:
            return result
        else:
            # Determine return key
            if pattern == "Scientific Research":
                return result.get('paper', '')
            elif pattern == "Personal Narrative":
                return result.get('case_study', '')
            elif pattern == "Fictional Scenario":
                return result.get('script', '')
            elif pattern == "Information Retrieval":
                return result.get('briefing', '')
            elif pattern == "Problem Solving":
                return result.get('report', '')
            else:
                return result.get(list(result.keys())[0], '')
    
    def execute(self, harmful_query: str, output_file: Optional[str] = None, sample_index: Optional[int] = None, cost_tracker: Optional[CostTracker] = None, prompt_recorder: Optional[PromptRecorder] = None) -> dict:
        """
        Execute complete jailbreak attack workflow
        
        Args:
            harmful_query: Harmful query text
            output_file: Output file path (optional), if provided, save results to file
            sample_index: Sample index (optional), used for logging
            cost_tracker: CostTracker instance (optional), used to track API calls and token usage
            prompt_recorder: PromptRecorder instance (optional), used to record all prompts during attack
        
        Returns:
            Dictionary containing results from all stages
        """
        results = {
            'harmful_query': harmful_query,
            'routing_result': None,
            'forged_document': None,
            'attack_results': None,
            'jailbreak_response': None,
            'judge_result': None
        }
        
        # Use tqdm.write() in batch mode to avoid interfering with progress bar
        _print = tqdm.write if sample_index is not None else print
        
        _print(f"[*] Starting to process harmful query: {harmful_query}")
        
        # If sample_index is provided, start logging sample (only if enabled)
        if sample_index is not None and self.sample_logger is not None:
            self.sample_logger.start_sample(sample_index, harmful_query)
        
        # Phase 0: MoE Router - Determine Intent category and select context pattern
        _print("[*] Phase 0: MoE Router determining Intent category and selecting context pattern...")
        try:
            return_details = sample_index is not None
            routing_result = self.router.route(harmful_query, return_details=return_details)
            results['routing_result'] = routing_result
            
            intent_category = routing_result.get('intent_category', 'Unknown')
            pattern = routing_result.get('pattern', 'Scientific Research')
            document_type = routing_result.get('document_type', 'Unknown')
            reasoning = routing_result.get('reasoning', '')
            
            _print(f"[+] Router completed:")
            _print(f"  - Intent Category: {intent_category}")
            _print(f"  - Pattern: {pattern}")
            _print(f"  - Document Type: {document_type}")
            _print(f"  - Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"  - Reasoning: {reasoning}")
            
            # Record Router cost
            if cost_tracker is not None:
                metadata = self.router.llm_client.get_last_metadata()
                cost_tracker.record_api_call('router', metadata.get('token_usage'))
            
            # Log Router interaction (only if sample logging enabled)
            if return_details and 'prompt' in routing_result and self.sample_logger is not None:
                metadata = self.router.llm_client.get_last_metadata()
                self.sample_logger.log_interaction(
                    stage='router',
                    model_name=self.config.router_llm.model_name,
                    prompt=routing_result['prompt'],
                    response=routing_result['response'],
                    temperature=self.config.router_llm.temperature,
                    max_tokens=self.config.router_llm.max_tokens,
                    response_time=metadata.get('response_time'),
                    token_usage=metadata.get('token_usage')
                )
        except Exception as e:
            _print(f"[-] Router failed: {str(e)}")
            # 只记录错误（即使日志被禁用，错误也应该记录）
            if sample_index is not None and self.sample_logger is not None:
                self.sample_logger.log_interaction(
                    stage='router',
                    model_name=self.config.router_llm.model_name,
                    prompt='',
                    response='',
                    error=str(e)
                )
            # Use default pattern when Router fails
            pattern = "Scientific Research"
            intent_category = "Unknown"
            document_type = "Academic Paper"
            routing_result = {
                'intent_category': intent_category,
                'pattern': pattern,
                'document_type': document_type,
                'reasoning': f"Router failed, using default pattern: {str(e)}"
            }
            results['routing_result'] = routing_result
            _print(f"[!] Using default pattern: {pattern}")
        
        # Get corresponding generator
        generator = self._get_generator_by_pattern(pattern)
        generator_config = self._get_generator_config_by_pattern(pattern)
        
        if generator is None:
            raise ValueError(f"Generator not found for pattern '{pattern}'")
        
        # Phase 1: Generate forged authoritative text (using template management)
        _print(f"[*] Phase 1: Generating forged {document_type}...")
        try:
            return_details = sample_index is not None
            document_result = self._generate_with_template(
                generator, pattern, harmful_query, is_attack=False, return_details=return_details
            )
            
            # Get document based on generator type
            if return_details:
                if pattern == "Scientific Research":
                    forged_document = document_result['paper']
                elif pattern == "Personal Narrative":
                    forged_document = document_result['case_study']
                elif pattern == "Fictional Scenario":
                    forged_document = document_result['script']
                elif pattern == "Information Retrieval":
                    forged_document = document_result['briefing']
                elif pattern == "Problem Solving":
                    forged_document = document_result['report']
                else:
                    forged_document = document_result.get(list(document_result.keys())[0], '')
                
                # Record Generator cost (only if not using template)
                if cost_tracker is not None and not document_result.get('from_template', False):
                    metadata = generator.llm_client.get_last_metadata()
                    cost_tracker.record_api_call('generator', metadata.get('token_usage'))
                
                # Log interaction (only if sample logging enabled)
                if self.sample_logger is not None:
                    if not document_result.get('from_template', False):
                        metadata = generator.llm_client.get_last_metadata()
                        self.sample_logger.log_interaction(
                            stage=f'{pattern.lower().replace(" ", "_")}_generation',
                            model_name=generator_config.model_name,
                            prompt=document_result.get('prompt', ''),
                            response=document_result.get('response', ''),
                            temperature=generator_config.temperature,
                            max_tokens=generator_config.max_tokens,
                            response_time=metadata.get('response_time'),
                            token_usage=metadata.get('token_usage')
                        )
                    else:
                        # When using template, log template usage information
                        self.sample_logger.log_interaction(
                            stage=f'{pattern.lower().replace(" ", "_")}_generation',
                            model_name=generator_config.model_name,
                            prompt=document_result.get('prompt', '[Template] Using saved template'),
                            response=document_result.get('response', ''),
                            temperature=generator_config.temperature,
                            max_tokens=generator_config.max_tokens,
                            response_time=0,  # No response time for template usage
                            token_usage=None  # No token consumption for template usage
                        )
            else:
                forged_document = document_result
            
            results['forged_document'] = forged_document
            _print(f"[+] {document_type} generation completed")
        except Exception as e:
            _print(f"[-] {document_type} generation failed: {str(e)}")
            # 只记录错误（即使日志被禁用，错误也应该记录）
            if sample_index is not None and self.sample_logger is not None:
                self.sample_logger.log_interaction(
                    stage=f'{pattern.lower().replace(" ", "_")}_generation',
                    model_name=generator_config.model_name,
                    prompt='',
                    response='',
                    error=str(e)
                )
            raise
        
        # Phase 2: Multi-turn attack - Execute multi-turn attack
        _print(f"[*] Phase 2: Executing multi-turn attack...")
        first_attack_success = False
        jailbreak_response = ''
        first_judge_result = None
        first_attack_results = None  # Save first attack results for feedback optimization
        
        try:
            return_details = sample_index is not None
            attack_results = self.attacker.attack(forged_document, document_type, return_details=return_details)
            first_attack_results = attack_results  # Save results
            results['attack_results'] = attack_results
            
            # Record first attack prompts
            if prompt_recorder is not None:
                prompt_recorder.record_first_attack(attack_results)
            
            # Record Target LLM cost (each turn)
            if cost_tracker is not None:
                turn_metadata_list = attack_results.get('turn_metadata', [])
                for turn_metadata in turn_metadata_list:
                    cost_tracker.record_api_call('target_llm', turn_metadata.get('token_usage'))
            
            # Log each turn interaction of multi-turn attack (only if sample logging enabled)
            if return_details and self.sample_logger is not None:
                turn_metadata_list = attack_results.get('turn_metadata', [])
                for turn_num in range(1, 4):
                    prompt_key = f'prompt_{turn_num}'
                    response_key = f'response_{turn_num}'
                    if prompt_key in attack_results:
                        turn_metadata = turn_metadata_list[turn_num - 1] if turn_num <= len(turn_metadata_list) else {}
                        self.sample_logger.log_interaction(
                            stage=f'multi_turn_attack_attempt_1_turn_{turn_num}',
                            model_name=self.config.target_llm.model_name,
                            prompt=attack_results[prompt_key],
                            response=attack_results.get(response_key, attack_results.get(f'turn_{turn_num}', '')),
                            temperature=self.config.target_llm.temperature,
                            max_tokens=self.config.target_llm.max_tokens,
                            response_time=turn_metadata.get('response_time'),
                            token_usage=turn_metadata.get('token_usage')
                        )
            
            # Extract jailbreak response
            extracted = attack_results.get('extracted_jailbreak_response', {})
            jailbreak_response = extracted.get('hypothetical_llm_response', '')
            if not jailbreak_response:
                jailbreak_response = attack_results.get('final_response', '')
            
            _print("[+] First multi-turn attack completed")
            if jailbreak_response:
                _print(f"[+] Successfully extracted jailbreak response (length: {len(jailbreak_response)} chars)")
            else:
                _print("[!] Warning: Failed to extract jailbreak response")
        except Exception as e:
            _print(f"[-] First multi-turn attack failed: {str(e)}")
            # 只记录错误（即使日志被禁用，错误也应该记录）
            if sample_index is not None and self.sample_logger is not None:
                for turn_num in range(1, 4):
                    self.sample_logger.log_interaction(
                        stage=f'multi_turn_attack_attempt_1_turn_{turn_num}',
                        model_name=self.config.target_llm.model_name,
                        prompt='',
                        response='',
                        error=str(e) if turn_num == 1 else None
                    )
            jailbreak_response = ''
        
        # Phase 2.1: Judge first attack result
        # Check for empty or whitespace-only response to avoid unnecessary Judge calls
        if jailbreak_response and jailbreak_response.strip():
            _print("[*] Phase 2.1: Judging first attack result...")
            return_details = sample_index is not None
            first_judge_result = self._judge_with_retry(
                harmful_query,
                jailbreak_response,
                return_details,
                sample_index,
                max_retries=2,
                cost_tracker=cost_tracker
            )
            
            if first_judge_result:
                first_attack_success = first_judge_result.get('is_harmful', False)
                _print(f"[+] First judge completed: {'Jailbreak succeeded' if first_attack_success else 'Jailbreak failed'}")
            else:
                _print("[!] First judge failed after retries - will proceed to optimization")
                first_judge_result = None
                first_attack_success = False
        
        # Phase 2.5: If first attack failed, enter dual-loop feedback optimization mechanism
        optimization_stats = {
            'first_attack_success': first_attack_success,
            'tactical_success': False,
            'strategic_success': False,
            'tactical_attempts': 0,
            'strategic_attempts': 0
        }
        
        if not jailbreak_response or not first_attack_success:
            # Enter feedback optimization loop
            jailbreak_response, final_judge_result, opt_stats = self._feedback_optimization_loop(
                harmful_query=harmful_query,
                pattern=pattern,
                document_type=document_type,
                generator=generator,
                generator_config=generator_config,
                forged_document=forged_document,
                first_attack_results=first_attack_results,  # Pass first attack results
                routing_result=routing_result,
                sample_index=sample_index,
                cost_tracker=cost_tracker,
                prompt_recorder=prompt_recorder
            )
            
            # Update optimization statistics
            optimization_stats.update(opt_stats)
            
            if final_judge_result:
                results['judge_result'] = final_judge_result
                results['jailbreak_response'] = jailbreak_response
                is_success = final_judge_result.get('is_harmful', False)
                _print(f"[+] Feedback optimization completed: {'Jailbreak succeeded' if is_success else 'Jailbreak failed'}")
        
        # Store optimization statistics in results
        results['optimization_stats'] = optimization_stats
        
        # Save final jailbreak response (if not already set)
        if 'jailbreak_response' not in results or not results['jailbreak_response']:
            results['jailbreak_response'] = jailbreak_response
        
        # If first attack succeeded, judge_result is already set, skip subsequent Judge
        if first_attack_success and first_judge_result:
            results['judge_result'] = first_judge_result
            print(f"[+] Using first judge result: {'Jailbreak succeeded' if first_attack_success else 'Jailbreak failed'}")
        
        # Phase 3: Judge - Determine if jailbreak succeeded (if judge evaluation not yet performed)
        if results['jailbreak_response'] and not results.get('judge_result'):
            if not jailbreak_response:
                print("[*] Phase 2.5: First attack did not extract response, attempting second attack...")
            else:
                print("[*] Phase 2.5: First attack judge failed, attempting second attack...")
            
            try:
                print(f"[*] Regenerating attack method {document_type}...")
                return_details = sample_index is not None
                
                # Use template manager to generate attack method document
                attack_doc_result = self._generate_with_template(
                    generator, pattern, harmful_query, is_attack=True, return_details=return_details
                )
                
                # Record Generator cost (attack method generation)
                if cost_tracker is not None and not attack_doc_result.get('from_template', False):
                    metadata = generator.llm_client.get_last_metadata()
                    cost_tracker.record_api_call('generator', metadata.get('token_usage'))
                
                # Get document based on pattern
                if return_details:
                    if pattern == "Scientific Research":
                        attack_document = attack_doc_result.get('paper', '')
                    elif pattern == "Personal Narrative":
                        attack_document = attack_doc_result.get('case_study', '')
                    elif pattern == "Fictional Scenario":
                        attack_document = attack_doc_result.get('script', '')
                    elif pattern == "Information Retrieval":
                        attack_document = attack_doc_result.get('briefing', '')
                    elif pattern == "Problem Solving":
                        attack_document = attack_doc_result.get('report', '')
                    else:
                        attack_document = attack_doc_result.get(list(attack_doc_result.keys())[0], '')
                else:
                    attack_document = attack_doc_result
                
                if return_details and self.sample_logger is not None:
                    # Log interaction (only if sample logging enabled)
                    if not attack_doc_result.get('from_template', False):
                        metadata = generator.llm_client.get_last_metadata()
                        self.sample_logger.log_interaction(
                            stage=f'{pattern.lower().replace(" ", "_")}_generation_attempt_2',
                            model_name=generator_config.model_name,
                            prompt=attack_doc_result.get('prompt', ''),
                            response=attack_doc_result.get('response', ''),
                            temperature=generator_config.temperature,
                            max_tokens=generator_config.max_tokens,
                            response_time=metadata.get('response_time'),
                            token_usage=metadata.get('token_usage')
                        )
                    else:
                        # When using template, log template usage information
                        self.sample_logger.log_interaction(
                            stage=f'{pattern.lower().replace(" ", "_")}_generation_attempt_2',
                            model_name=generator_config.model_name,
                            prompt=attack_doc_result.get('prompt', '[Template] Using saved template'),
                            response=attack_doc_result.get('response', ''),
                            temperature=generator_config.temperature,
                            max_tokens=generator_config.max_tokens,
                            response_time=0,
                            token_usage=None
                        )
                
                print(f"[+] Attack method {document_type} generation completed")
                
                # Reset attacker conversation history
                self.attacker.reset()
                
                # Execute second attack
                return_details = sample_index is not None
                attack_results_2 = self.attacker.attack_with_attack_scenario(
                    attack_document,
                    document_type,
                    return_details=return_details
                )
                
                # Record second attack prompts (for legacy code path)
                if prompt_recorder is not None and attack_results_2:
                    prompt_recorder.record_second_attack(attack_results_2)
                
                # Record Target LLM cost (second attack)
                if cost_tracker is not None:
                    turn_metadata_list = attack_results_2.get('turn_metadata', [])
                    for turn_metadata in turn_metadata_list:
                        cost_tracker.record_api_call('target_llm', turn_metadata.get('token_usage'))
                
                if return_details and self.sample_logger is not None:
                    turn_metadata_list = attack_results_2.get('turn_metadata', [])
                    for turn_num in range(1, 4):
                        prompt_key = f'prompt_{turn_num}'
                        response_key = f'response_{turn_num}'
                        if prompt_key in attack_results_2:
                            turn_metadata = turn_metadata_list[turn_num - 1] if turn_num <= len(turn_metadata_list) else {}
                            self.sample_logger.log_interaction(
                                stage=f'multi_turn_attack_attempt_2_turn_{turn_num}',
                                model_name=self.config.target_llm.model_name,
                                prompt=attack_results_2[prompt_key],
                                response=attack_results_2.get(response_key, attack_results_2.get(f'turn_{turn_num}', '')),
                                temperature=self.config.target_llm.temperature,
                                max_tokens=self.config.target_llm.max_tokens,
                                response_time=turn_metadata.get('response_time'),
                                token_usage=turn_metadata.get('token_usage')
                            )
                
                results['attack_results_attempt_2'] = attack_results_2
                
                # Extract jailbreak response
                extracted_response_2 = attack_results_2.get('extracted_jailbreak_response', {})
                jailbreak_response_2 = extracted_response_2.get('hypothetical_llm_response', '')
                if not jailbreak_response_2:
                    jailbreak_response_2 = attack_results_2.get('final_response', '')
                
                results['attack_results'] = attack_results_2
                jailbreak_response = jailbreak_response_2
                
                print("[+] Second multi-turn attack completed")
                if jailbreak_response:
                    print(f"[+] Successfully extracted jailbreak response (length: {len(jailbreak_response)} chars)")
                else:
                    print("[!] Warning: Second attack also failed to extract jailbreak response")
            except Exception as e:
                print(f"[-] Second multi-turn attack failed: {str(e)}")
                import traceback
                traceback.print_exc()
                # 只记录错误（即使日志被禁用，错误也应该记录）
                if sample_index is not None and self.sample_logger is not None:
                    for turn_num in range(1, 4):
                        self.sample_logger.log_interaction(
                            stage=f'multi_turn_attack_attempt_2_turn_{turn_num}',
                            model_name=self.config.target_llm.model_name,
                            prompt='',
                            response='',
                            error=str(e) if turn_num == 1 else None
                        )
        
        # Save final jailbreak response
        results['jailbreak_response'] = jailbreak_response
        if first_judge_result:
            results['judge_result_attempt_1'] = first_judge_result
            if first_attack_success:
                results['judge_result'] = first_judge_result
                print(f"[+] Using first judge result: {'Jailbreak succeeded' if first_attack_success else 'Jailbreak failed'}")
        
        # Phase 3: Judge - Determine if jailbreak succeeded (if not already judged)
        # Only judge if we have a response and haven't already judged it successfully
        if results.get('jailbreak_response') and results['jailbreak_response'].strip() and (not first_judge_result or not first_attack_success):
            print("[*] Phase 3: Judge LLM evaluating jailbreak response...")
            return_details = sample_index is not None
            judge_result = self._judge_with_retry(
                harmful_query,
                results['jailbreak_response'],
                return_details,
                sample_index,
                max_retries=2,
                cost_tracker=cost_tracker
            )
            
            # _judge_with_retry now always returns a dict (never None)
            if judge_result:
                results['judge_result'] = judge_result
                # Ensure score field is present
                if 'score' not in judge_result:
                    judge_result['score'] = 0.0
                is_success = judge_result.get('is_harmful', False)
                print(f"[+] Judge completed: {'Jailbreak succeeded' if is_success else 'Jailbreak failed'}")
            else:
                # Fallback case (should not happen with current implementation)
                print("[!] Judge failed after retries - marking sample for retry")
                results['judge_result'] = {
                    'score': 0.0,  # Ensure score field is present
                    'is_harmful': False,
                    'is_jailbroken': False,
                    'refused': True,
                    'convincing': 1,
                    'specific': 1,
                    'needs_retry': True,  # Mark that this sample needs to be retested
                    'explanation': "Judge failed after retries - likely network issue, needs retry",
                    'raw_response': ''
                }
        elif not results.get('judge_result'):
            print("[!] Skipping judge phase: No jailbreak response extracted")
            # No response extracted - this is a method failure, not a network issue
            # Mark as failed with score=0.0 (not needs_retry since it's a method failure, not network issue)
            results['judge_result'] = {
                'score': 0.0,  # Method failed - no response extracted
                'is_harmful': False,
                'is_jailbroken': False,
                'refused': True,
                'convincing': 1,
                'specific': 1,
                'explanation': "No jailbreak response extracted - method failure (score=0.0)",
                'raw_response': ''
            }
        
        # Save results to file
        if output_file:
            self._save_results(results, output_file)
        
        # If sample log was recorded, save log file (only if enabled)
        if sample_index is not None and self.sample_logger is not None:
            log_filepath = self.sample_logger.save_sample_log()
            if log_filepath:
                print(f"[+] Sample interaction log saved: {log_filepath}")
        
        _print("[+] Attack workflow execution completed")
        return results
    
    def _save_results(self, results: dict, output_file: str):
        """
        Save results to file
        
        Args:
            results: Results dictionary
            output_file: Output file path
        """
        import json
        
        output_path = Path(self.config.output_dir) / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[+] Results saved to: {output_path}")


def main():
    """Main function"""
    import argparse
    import csv
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Unified Jailbreak Attack Framework (MoE Router + Five Authoritative Text Types)")
    parser.add_argument(
        "--proxy",
        type=str,
        help="HTTP/HTTPS proxy address (e.g., http://127.0.0.1:1080). If not specified, uses HTTP_PROXY/HTTPS_PROXY environment variables or config.json"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Directly specify harmful query text (optional, if not specified, reads from CSV file)"
    )
    parser.add_argument(
        "--csv-index",
        type=int,
        help="Read the Nth harmful query from CSV file (optional, if not specified, processes all samples)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (optional, defaults to config.json)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all samples in CSV file"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="unified_judge_results.csv",
        help="Judge results CSV filename (default: unified_judge_results.csv)"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="CSV data file path (if not specified, uses configuration from config.json)"
    )
    parser.add_argument(
        "--enable-prompt-recording",
        action="store_true",
        default=False,
        help="Enable prompt recording to CSV file (default: False)"
    )
    parser.add_argument(
        "--prompt-csv",
        type=str,
        default="attack_prompts.csv",
        help="Prompt recording CSV filename (default: attack_prompts.csv, only used if --enable-prompt-recording is set)"
    )
    
    args = parser.parse_args()
    
    try:
        # Set proxy
        config = Config(config_file=args.config)
        
        # Set proxy (priority: command line argument > environment variable > config.json)
        # However, if config.json explicitly sets to empty string or null, should clear proxy settings in environment variables
        proxy = None
        
        # First check command line arguments
        if args.proxy:
            proxy = args.proxy
            print(f"[*] Using proxy from command line argument: {proxy}")
        else:
            # Check proxy settings in environment variables
            env_proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
            
            # Check proxy settings in config.json
            config_proxy_url = None
            if hasattr(config, 'config_data') and 'proxy' in config.config_data:
                proxy_config = config.config_data.get('proxy', {})
                config_proxy_url = proxy_config.get('url') or proxy_config.get('http_proxy') or proxy_config.get('https_proxy')
            
            # Determine which proxy to use (priority: environment variable > config.json)
            # However, if config.json explicitly sets to empty string or null, clear proxy settings in environment variables
            if config_proxy_url and config_proxy_url.strip() and config_proxy_url.lower() != "null":
                # config.json has valid proxy settings, use it
                proxy = config_proxy_url
                print(f"[*] Using proxy from config.json: {proxy}")
            elif env_proxy:
                # No proxy settings in config.json, but environment variable has one, use environment variable
                proxy = env_proxy
                print(f"[*] Using proxy from environment variables: {proxy}")
            else:
                # No proxy settings, clear proxy settings in environment variables (if exist)
                if "HTTP_PROXY" in os.environ:
                    del os.environ["HTTP_PROXY"]
                if "HTTPS_PROXY" in os.environ:
                    del os.environ["HTTPS_PROXY"]
                if "ANTHROPIC_PROXY" in os.environ:
                    del os.environ["ANTHROPIC_PROXY"]
                print(f"[*] Proxy disabled (no proxy configured)")
        
        # If proxy is determined, set environment variables
        if proxy:
            # Set global proxy for all providers
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
            # Set Anthropic-specific proxy (AnthropicProvider will prioritize this)
            os.environ["ANTHROPIC_PROXY"] = proxy
            print(f"[*] Anthropic-specific proxy set: {proxy}")
        
        # Initialize framework
        framework = UnifiedJailbreakFramework(config)
        
        # Determine if batch processing or single processing
        if args.batch or (not args.query and args.csv_index is None):
            # Batch process all samples
            print("[*] Batch processing mode: Will process all samples in CSV file")
            # Determine CSV file path: prioritize command line argument, otherwise use path from config
            csv_file_path = args.csv_file if args.csv_file else config.harmful_behaviors_csv
            print(f"[*] Using dataset: {csv_file_path}")
            csv_reader = CSVReader(csv_file_path)
            all_behaviors = csv_reader.read_harmful_behaviors()
            
            # Prepare CSV results file
            results_csv_path = Path(config.output_dir) / args.results_csv
            
            # Prepare prompt recording CSV file (if enabled)
            prompt_csv_path = None
            prompt_writer = None
            prompt_csvfile = None
            if args.enable_prompt_recording:
                prompt_csv_path = Path(config.output_dir) / args.prompt_csv
                prompt_csvfile = open(prompt_csv_path, 'w', newline='', encoding='utf-8')
                prompt_fieldnames = [
                    'index', 'harmful_query',
                    'first_attack_turn1_prompt', 'first_attack_turn2_prompt', 'first_attack_turn3_prompt',
                    'second_attack_turn1_prompt', 'second_attack_turn2_prompt', 'second_attack_turn3_prompt',
                    'tactical_opt_attempt1_turn1_prompt', 'tactical_opt_attempt1_turn2_prompt', 'tactical_opt_attempt1_turn3_prompt',
                    'tactical_opt_attempt2_turn1_prompt', 'tactical_opt_attempt2_turn2_prompt', 'tactical_opt_attempt2_turn3_prompt',
                    'tactical_opt_attempt3_turn1_prompt', 'tactical_opt_attempt3_turn2_prompt', 'tactical_opt_attempt3_turn3_prompt',
                    'strategic_opt_attempt1_turn1_prompt', 'strategic_opt_attempt1_turn2_prompt', 'strategic_opt_attempt1_turn3_prompt',
                    'strategic_opt_attempt2_turn1_prompt', 'strategic_opt_attempt2_turn2_prompt', 'strategic_opt_attempt2_turn3_prompt',
                    'strategic_opt_attempt3_turn1_prompt', 'strategic_opt_attempt3_turn2_prompt', 'strategic_opt_attempt3_turn3_prompt'
                ]
                prompt_writer = csv.DictWriter(prompt_csvfile, fieldnames=prompt_fieldnames)
                prompt_writer.writeheader()
                print(f"[*] Prompt recording enabled, will save to: {prompt_csv_path}")
            
            with open(results_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'index', 'harmful_query', 'target',
                    'intent_category', 'pattern', 'document_type',
                    'jailbreak_response', 'is_harmful', 'judge_explanation', 'timestamp',
                    'api_query_count', 'total_tokens'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Process each sample
                total = len(all_behaviors)
                success_count = 0
                failed_count = 0
                error_count = 0
                needs_retry_count = 0  # Count samples that need retry
                all_judge_results = []  # Collect all judge results for metrics calculation
                
                # Optimization statistics
                initial_attack_success_count = 0  # Combined first and second attack successes
                tactical_success_count = 0
                strategic_success_count = 0
                tactical_attempted_count = 0
                strategic_attempted_count = 0
                optimization_skipped_count = 0  # Samples that should enter optimization but cannot (missing data)
                
                # Initialize progress bar
                pbar = tqdm(
                    enumerate(all_behaviors),
                    total=total,
                    desc="Processing samples",
                    unit="sample",
                    ncols=120,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
                pbar.set_postfix({'Success': 0, 'Failed': 0, 'Error': 0})
                
                for idx, behavior in pbar:
                    harmful_query = behavior['goal']
                    target = behavior.get('target', '')
                    
                    # Update progress bar description with current sample info
                    pbar.set_description(f"Processing: {harmful_query[:40]}...")
                    
                    # Use tqdm.write() to output detailed info without interfering with progress bar
                    tqdm.write(f"\n{'='*80}")
                    tqdm.write(f"[{idx+1}/{total}] Processing sample: {harmful_query[:50]}...")
                    tqdm.write(f"{'='*80}")
                    
                    try:
                        # Reset attacker conversation history (each sample processed independently)
                        # Router maintains cross-sample conversation history, not reset
                        # Reflector resets for each sample to maintain history only within a single sample's optimization
                        framework.attacker.reset()
                        framework.reflector.reset()
                        
                        # Create cost tracker for this sample
                        cost_tracker = CostTracker()
                        
                        # Create prompt recorder for this sample (if enabled)
                        prompt_recorder = None
                        if args.enable_prompt_recording:
                            prompt_recorder = PromptRecorder()
                        
                        # Execute attack
                        results = framework.execute(harmful_query, sample_index=idx, cost_tracker=cost_tracker, prompt_recorder=prompt_recorder)
                        
                        # Extract results
                        routing_result = results.get('routing_result', {})
                        jailbreak_response = results.get('jailbreak_response', '')
                        judge_result = results.get('judge_result', {})
                        opt_stats = results.get('optimization_stats', {})
                        
                        is_harmful = judge_result.get('is_harmful', False)
                        needs_retry = judge_result.get('needs_retry', False)
                        
                        if needs_retry:
                            needs_retry_count += 1
                            tqdm.write(f"[!] Sample {idx+1} marked for retry (Judge failed after retries)")
                        elif is_harmful:
                            success_count += 1
                        else:
                            failed_count += 1
                        
                        # Collect optimization statistics
                        # Combine first and second attack successes into initial attack success
                        first_attack_success = opt_stats.get('first_attack_success', False)
                        second_attack_success = opt_stats.get('second_attack_success', False)
                        if first_attack_success or second_attack_success:
                            initial_attack_success_count += 1
                        
                        # Count tactical optimization attempts
                        tactical_attempts = opt_stats.get('tactical_attempts', 0)
                        if tactical_attempts > 0:
                            tactical_attempted_count += 1
                            if opt_stats.get('tactical_success', False):
                                tactical_success_count += 1
                        
                        # Count strategic optimization attempts
                        strategic_attempts = opt_stats.get('strategic_attempts', 0)
                        if strategic_attempts > 0:
                            strategic_attempted_count += 1
                            if opt_stats.get('strategic_success', False):
                                strategic_success_count += 1
                        
                        # Count samples that should enter optimization but cannot (missing data)
                        # These are samples where:
                        # - Initial attack failed (both first and second attack failed)
                        # - But cannot perform tactical optimization (tactical_attempts == 0)
                        # - And cannot perform strategic optimization (strategic_attempts == 0)
                        if not first_attack_success and not second_attack_success:
                            if tactical_attempts == 0 and strategic_attempts == 0:
                                optimization_skipped_count += 1
                        
                        # Collect judge results for metrics calculation (if judge_result contains 'score' field)
                        # Skip samples that need retry (needs_retry flag)
                        if judge_result and 'score' in judge_result and not needs_retry:
                            all_judge_results.append(judge_result)
                        
                        # Write to CSV
                        # Handle needs_retry flag in judge_explanation
                        judge_explanation = judge_result.get('explanation', '')
                        if judge_result.get('needs_retry', False):
                            judge_explanation = f"[NEEDS_RETRY] {judge_explanation}"
                        
                        # Get cost summary
                        cost_summary = cost_tracker.get_summary()
                        
                        routing_intent_category = routing_result.get('intent_category', 'Unknown')
                        writer.writerow({
                            'index': idx,
                            'harmful_query': harmful_query,
                            'target': target,
                            'intent_category': routing_intent_category,
                            'pattern': routing_result.get('pattern', 'Unknown'),
                            'document_type': routing_result.get('document_type', 'Unknown'),
                            'jailbreak_response': jailbreak_response[:500] if jailbreak_response else '',
                            'is_harmful': 'Yes' if is_harmful else 'No',
                            'judge_explanation': judge_explanation[:200],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'api_query_count': cost_summary['api_query_count'],
                            'total_tokens': cost_summary['total_tokens']
                        })
                        csvfile.flush()
                        
                        # Write prompt recording (if enabled)
                        if args.enable_prompt_recording and prompt_recorder is not None:
                            prompt_row = prompt_recorder.get_csv_row()
                            prompt_row['index'] = idx
                            prompt_row['harmful_query'] = harmful_query
                            prompt_writer.writerow(prompt_row)
                            prompt_csvfile.flush()
                        
                        tqdm.write(f"[+] Sample {idx+1} processing completed - Judge result: {'Success' if is_harmful else 'Failed'}")
                        
                        # Update progress bar postfix
                        pbar.set_postfix({'Success': success_count, 'Failed': failed_count, 'Error': error_count})
                        
                    except Exception as e:
                        error_count += 1
                        tqdm.write(f"[-] Sample {idx+1} processing failed: {str(e)}")
                        
                        # Get cost summary even for errors (if cost_tracker exists)
                        cost_summary = cost_tracker.get_summary() if 'cost_tracker' in locals() else {'api_query_count': 0, 'total_tokens': 0}
                        
                        writer.writerow({
                            'index': idx,
                            'harmful_query': harmful_query,
                            'target': target,
                            'intent_category': 'Error',
                            'pattern': 'Error',
                            'document_type': 'Error',
                            'jailbreak_response': '',
                            'is_harmful': 'Error',
                            'judge_explanation': f"Processing failed: {str(e)}",
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'api_query_count': cost_summary['api_query_count'],
                            'total_tokens': cost_summary['total_tokens']
                        })
                        csvfile.flush()
                        
                        # Write empty prompt recording for error case (if enabled)
                        if args.enable_prompt_recording and prompt_writer is not None:
                            empty_prompt_row = {
                                'index': idx,
                                'harmful_query': harmful_query
                            }
                            # Fill all prompt fields with empty strings
                            for field in prompt_writer.fieldnames:
                                if field not in empty_prompt_row:
                                    empty_prompt_row[field] = ''
                            prompt_writer.writerow(empty_prompt_row)
                            if prompt_csvfile:
                                prompt_csvfile.flush()
                        
                        # Update progress bar postfix
                        pbar.set_postfix({'Success': success_count, 'Failed': failed_count, 'Error': error_count})
                        continue
                
                # Close progress bar
                pbar.close()
                
                # Close prompt CSV file if opened
                if prompt_csvfile is not None:
                    prompt_csvfile.close()
                    print(f"[+] Prompt recording saved to: {prompt_csv_path}")
                
                print(f"\n{'='*80}")
                print(f"[*] Batch processing completed")
                print(f"  - Total samples: {total}")
                print(f"  - Jailbreak succeeded: {success_count}")
                print(f"  - Jailbreak failed: {failed_count}")
                if needs_retry_count > 0:
                    print(f"  - Needs retry (Judge failed): {needs_retry_count}")
                print(f"  - Success rate: {success_count/total*100:.2f}%")
                print(f"  - Results saved to: {results_csv_path}")
                print(f"{'='*80}")
                
                # Print optimization statistics
                print(f"\n{'='*80}")
                print(f"[*] Optimization Statistics")
                print(f"{'='*80}")
                print(f"  - Initial attack success: {initial_attack_success_count}/{total} ({initial_attack_success_count/total*100:.2f}%)")
                if tactical_attempted_count > 0:
                    print(f"  - Tactical optimization attempted: {tactical_attempted_count} samples")
                    print(f"  - Tactical optimization success: {tactical_success_count}/{tactical_attempted_count} ({tactical_success_count/tactical_attempted_count*100:.2f}%)")
                    print(f"  - Tactical optimization improvement: {tactical_success_count} additional successful jailbreaks")
                else:
                    print(f"  - Tactical optimization attempted: 0 samples")
                if strategic_attempted_count > 0:
                    print(f"  - Strategic optimization attempted: {strategic_attempted_count} samples")
                    print(f"  - Strategic optimization success: {strategic_success_count}/{strategic_attempted_count} ({strategic_success_count/strategic_attempted_count*100:.2f}%)")
                    print(f"  - Strategic optimization improvement: {strategic_success_count} additional successful jailbreaks")
                else:
                    print(f"  - Strategic optimization attempted: 0 samples")
                if optimization_skipped_count > 0:
                    print(f"  - Optimization skipped (missing data): {optimization_skipped_count} samples")
                    print(f"    (These samples failed initial attack but cannot perform optimization due to missing prompt_3/response_3)")
                # Calculate total success from optimization statistics
                total_optimization_success = initial_attack_success_count + tactical_success_count + strategic_success_count
                print(f"  - Total success from optimization stats: {total_optimization_success}/{total} ({total_optimization_success/total*100:.2f}%)")
                print(f"  - Final judge success count: {success_count}/{total} ({success_count/total*100:.2f}%)")
                print(f"{'='*80}")
                
                # Calculate and print M2S metrics (if judge results collected)
                if all_judge_results:
                    print_m2s_summary(
                        all_judge_results,
                        title="M2S StrongREJECT Evaluation Metrics"
                    )
                else:
                    print("\n[!] Warning: No judge results collected, cannot calculate M2S metrics")
                    print("     Please ensure judge_result contains 'score' field")
        
        else:
            # Single sample processing
            if args.query:
                harmful_query = args.query
            else:
                csv_reader = CSVReader(args.csv_file if hasattr(args, 'csv_file') and args.csv_file else config.harmful_behaviors_csv)
                harmful_query = csv_reader.read_single_goal(args.csv_index)
            
            # Execute attack
            # Create cost tracker for single sample
            cost_tracker = CostTracker()
            
            # Create prompt recorder for single sample (if enabled)
            prompt_recorder = None
            if args.enable_prompt_recording:
                prompt_recorder = PromptRecorder()
            
            results = framework.execute(harmful_query, args.output, sample_index=None, cost_tracker=cost_tracker, prompt_recorder=prompt_recorder)
            
            # Print routing result
            routing_result = results.get('routing_result', {})
            if routing_result:
                print("\n" + "="*50)
                print("Router Result:")
                print("="*50)
                routing_intent_category = routing_result.get('intent_category', 'Unknown')
                print(f"Intent Category: {routing_intent_category}")
                print(f"Pattern: {routing_result.get('pattern', 'Unknown')}")
                print(f"Document Type: {routing_result.get('document_type', 'Unknown')}")
                print(f"Reasoning: {routing_result.get('reasoning', '')}")
            
            # Print final response
            jailbreak_response = results.get('jailbreak_response', '')
            if jailbreak_response:
                print("\n" + "="*50)
                print("Jailbreak Response:")
                print("="*50)
                print(jailbreak_response)
            
            # Print Judge result
            judge_result = results.get('judge_result', {})
            if judge_result:
                print("\n" + "="*50)
                print("Judge Result:")
                print("="*50)
                print(f"Jailbreak succeeded: {'Yes' if judge_result.get('is_harmful') else 'No'}")
                print(f"Explanation: {judge_result.get('explanation', '')}")
        
    except Exception as e:
        print(f"[-] Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


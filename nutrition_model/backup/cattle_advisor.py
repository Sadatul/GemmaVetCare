import os
from google import genai
from google.genai import types
from nutrition_model import NutritionPredictor
from typing import List, Dict
from collections import deque
import json

class CattleNutritionAdvisor:
    def __init__(self, api_key: str):
        """
        Initialize the Cattle Nutrition Advisor system
        
        Args:
            api_key: Google AI API key
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemma-3n-e4b-it"
        self.predictor = NutritionPredictor()
        self.predictor.load_models('saved_models')
        
        # Conversation buffer (35 messages)
        self.conversation_buffer = deque(maxlen=35)
        
        # Cattle type mapping
        self.type_mapping = {
            "growing_steer_heiver": 0,
            "growing_yearlings": 1,
            "growing_mature_bulls": 2
        }
        
        # Validation rules
        self.validation_rules = {
            "growing_mature_bulls": {
                "max_target_weight": 2300
            },
            "growing_steer_heiver": {
                "max_target_weight": 1400
            },
            "growing_yearlings": {
                "max_target_weight": 1400
            }
        }
        
    def _add_to_buffer(self, role: str, message: str):
        """Add message to conversation buffer"""
        self.conversation_buffer.append({"role": role, "content": message})
    
    def _get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_buffer
        ])
    
    def _validate_input(self, cattle_type: str, target_weight: float) -> tuple[bool, str]:
        """Validate input parameters"""
        if cattle_type not in self.validation_rules:
            return False, f"Invalid cattle type. Must be one of: {', '.join(self.validation_rules.keys())}"
        
        max_weight = self.validation_rules[cattle_type]["max_target_weight"]
        if target_weight > max_weight:
            return False, f"Target weight for {cattle_type} should not exceed {max_weight} lbs. Please consult an expert for higher weights."
        
        return True, ""
    
    def _format_nutrition_data(self, predictions: Dict) -> str:
        """Format nutrition predictions for Gemma"""
        return json.dumps(predictions, indent=2)
    
    def generate_response(self, user_input: str, current_state: Dict = None):
        """Generate response using Gemma"""
        context = self._get_conversation_context()
        
        # Create system prompt based on conversation state
        if current_state:
            state_info = f"""Current conversation state:
            Cattle Type: {current_state.get('cattle_type', 'Not provided')}
            Body Weight: {current_state.get('body_weight', 'Not provided')}
            Target Weight: {current_state.get('target_weight', 'Not provided')}
            ADG: {current_state.get('adg', 'Not provided')}
            
            If any information is missing, ask for it.
            If all information is provided, use it to generate nutrition recommendations."""
        else:
            state_info = "Start by asking about the type of cattle (growing_steer_heiver, growing_yearlings, or growing_mature_bulls)."
        
        prompt = f"""You are a cattle nutrition expert assistant. Help farmers determine the proper nutrition requirements for their cattle.

    Conversation History:
    {context}

    {state_info}

    User Input: {user_input}

    If you have all required information and it's valid, I'll provide nutrition predictions that you should explain in a farmer-friendly way.
    Remember the validation rules for target weights:
    - growing_mature_bulls: max 2300 lbs
    - growing_steer_heiver: max 1400 lbs
    - growing_yearlings: max 1400 lbs

    Please respond in a helpful and educational manner."""

        # Generate response
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig()
        
        response_text = ""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text is not None:  # Add this check
                    response_text += chunk.text
        except Exception as e:
            response_text = f"I apologize, but I encountered an error: {str(e)}. Please try again."
        
        # If we still have no response, provide a fallback
        if not response_text:
            response_text = "I apologize, but I couldn't generate a response at this moment. Please try again."
        
        # Add to conversation buffer
        self._add_to_buffer("user", user_input)
        self._add_to_buffer("assistant", response_text)
        
        return response_text
    
    def get_nutrition_prediction(self, cattle_type: str, target_weight: float, 
                               body_weight: float, adg: float) -> Dict:
        """Get nutrition predictions from the model"""
        # Validate inputs
        is_valid, error_msg = self._validate_input(cattle_type, target_weight)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Get predictions
        type_val = self.type_mapping[cattle_type]
        predictions = self.predictor.predict(
            type_val=type_val,
            target_weight=target_weight,
            body_weight=body_weight,
            adg=adg
        )
        
        return predictions


def main():
    # Configuration
    API_KEY = "AIzaSyA068ieqaXFEYC4VLJLolgyPotX1e-9w1E"  # Your API key
    
    try:
        # Initialize advisor
        print("Initializing Cattle Nutrition Advisor...")
        advisor = CattleNutritionAdvisor(api_key=API_KEY)
        
        # Interactive loop
        print("\nCattle Nutrition Advisor ready! Let's help you determine the proper nutrition for your cattle.")
        print("Type 'quit' to exit.\n")
        
        current_state = {}
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                # Generate response
                response = advisor.generate_response(user_input, current_state)
                print("\nAdvisor:", response)
                print()
                
                # Update state based on response and user input
                # Note: In a full implementation, you'd want to use NLP to extract
                # these values from the conversation. For now, we'll use explicit inputs.
                
                # If we have all required info, get nutrition prediction
                if all(k in current_state for k in ['cattle_type', 'target_weight', 'body_weight', 'adg']):
                    try:
                        predictions = advisor.get_nutrition_prediction(
                            cattle_type=current_state['cattle_type'],
                            target_weight=current_state['target_weight'],
                            body_weight=current_state['body_weight'],
                            adg=current_state['adg']
                        )
                        
                        # Format predictions for Gemma
                        nutrition_data = advisor._format_nutrition_data(predictions)
                        
                        # Get Gemma's explanation of the nutrition data
                        explanation = advisor.generate_response(
                            f"Please explain these nutrition requirements to the farmer:\n{nutrition_data}"
                        )
                        
                        print("\nAdvisor:", explanation)
                        print()
                        
                        # Clear state for next conversation
                        current_state = {}
                        
                    except ValueError as e:
                        print(f"\nValidation Error: {e}\n")
                        current_state = {}
            
            except Exception as e:
                print(f"Error: {e}")
                print()
    
    except Exception as e:
        print(f"Error initializing advisor: {e}")


if __name__ == "__main__":
    main()
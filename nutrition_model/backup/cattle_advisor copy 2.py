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
        
        # Setup function calling tools
        self.nutrition_tool = types.Tool(function_declarations=[{
            "name": "get_nutrition_prediction",
            "description": "Get nutrition predictions for cattle based on their type, target weight, current body weight, and average daily gain (ADG).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cattle_type": {
                        "type": "string",
                        "enum": ["growing_steer_heiver", "growing_yearlings", "growing_mature_bulls"],
                        "description": "Type of cattle"
                    },
                    "target_weight": {
                        "type": "number",
                        "description": "Target weight for the cattle in pounds"
                    },
                    "body_weight": {
                        "type": "number",
                        "description": "Current body weight of the cattle in pounds"
                    },
                    "adg": {
                        "type": "number",
                        "description": "Average Daily Gain (ADG) in pounds"
                    }
                },
                "required": ["cattle_type", "target_weight", "body_weight", "adg"]
            }
        }])
        
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
        """Generate response using Gemma with function calling"""
        context = self._get_conversation_context()
        
        function_spec = """
# Function: get_nutrition_prediction
# Description: Get nutrition predictions for cattle based on their type, target weight, current body weight, and average daily gain (ADG).
#
# Parameters:
# - cattle_type (string): Type of cattle. Must be one of: growing_steer_heiver, growing_yearlings, growing_mature_bulls
# - target_weight (number): Target weight for the cattle in pounds
# - body_weight (number): Current body weight of the cattle in pounds
# - adg (number): Average Daily Gain (ADG) in pounds
#
# Returns: Dict with nutrition predictions
"""

        function_call_guide = """To call the function, respond in this exact format:
<function_call>
get_nutrition_prediction(
    cattle_type="type_here",
    target_weight=weight_here,
    body_weight=weight_here,
    adg=value_here
)
</function_call>"""
        
        # Create system prompt based on conversation state
        if current_state:
            state_info = f"""Current conversation state:
            Cattle Type: {current_state.get('cattle_type', 'Not provided')}
            Body Weight: {current_state.get('body_weight', 'Not provided')}
            Target Weight: {current_state.get('target_weight', 'Not provided')}
            ADG: {current_state.get('adg', 'Not provided')}
            
            If any information is missing, ask for it.
            If all information is provided, call the get_nutrition_prediction function."""
        else:
            state_info = "Start by asking about the type of cattle (growing_steer_heiver, growing_yearlings, or growing_mature_bulls)."
        
        prompt = f"""You are a cattle nutrition expert assistant. Help farmers determine the proper nutrition requirements for their cattle.

{function_spec}

{function_call_guide}

Conversation History:
{context}

{state_info}

User Input: {user_input}

When you have all required information and it's valid, call the get_nutrition_prediction function to get nutrition recommendations.
Remember the validation rules for target weights:
- growing_mature_bulls: max 2300 lbs
- growing_steer_heiver: max 1400 lbs
- growing_yearlings: max 1400 lbs

After getting predictions, explain them in a farmer-friendly way."""

        # Generate initial response
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )]
            )
            
            response_text = response.text
            
            # Check for function call
            if "<function_call>" in response_text and "</function_call>" in response_text:
                # Extract function call
                function_call_text = response_text[
                    response_text.find("<function_call>") + len("<function_call>"):
                    response_text.find("</function_call>")
                ].strip()
                
                # Parse function call (simple parsing since we know the format)
                try:
                    # Extract parameters from the function call text
                    params_text = function_call_text[function_call_text.find("(") + 1:function_call_text.rfind(")")]
                    params_dict = {}
                    
                    for param in params_text.split(","):
                        if "=" in param:
                            key, value = param.strip().split("=")
                            key = key.strip()
                            value = value.strip().strip('"')
                            if value.replace(".", "").isdigit():
                                value = float(value)
                            params_dict[key] = value
                    
                    # Call the function with parsed parameters
                    predictions = self.get_nutrition_prediction(**params_dict)
                    
                    # Get explanation for the predictions
                    explanation_prompt = f"""Here are the nutrition predictions:
                    {json.dumps(predictions, indent=2)}
                    
                    Please explain these recommendations in a farmer-friendly way, including specific details about 
                    dry matter intake, protein requirements, and energy needs."""
                    
                    explanation_response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=explanation_prompt)]
                        )]
                    )
                    
                    response_text = explanation_response.text
                    
                except (ValueError, TypeError) as e:
                    response_text = f"I apologize, but there was an error with the function parameters: {str(e)}"
                except Exception as e:
                    response_text = f"I apologize, but there was an error calling the function: {str(e)}"
            else:
                # No function call found, use the original response
                response_text = response.text
                
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
        print("\nCattle Nutrition Advisor ready! Let's help you determine the proper nutrition for your cattle. with model: " + advisor.model_name)
        print("Type 'quit' to exit.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                # Generate response using function calling
                response = advisor.generate_response(user_input)
                print("\nAdvisor:", response)
                print()
            
            except Exception as e:
                print(f"Error: {e}")
                print()
    
    except Exception as e:
        print(f"Error initializing advisor: {e}")


if __name__ == "__main__":
    main()
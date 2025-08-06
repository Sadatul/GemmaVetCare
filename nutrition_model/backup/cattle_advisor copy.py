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
        self.model_name = "gemini-2.5-flash"
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
            If all information is provided, call the get_nutrition_prediction function."""
        else:
            state_info = "Start by asking about the type of cattle (growing_steer_heiver, growing_yearlings, or growing_mature_bulls)."
        
        prompt = f"""You are a cattle nutrition expert assistant. Help farmers determine the proper nutrition requirements for their cattle.

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

        # Generate response
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        # Add tool configuration
        config = types.GenerateContentConfig(
            tools=[self.nutrition_tool]
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Check for function call
            if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                
                if function_call.name == "get_nutrition_prediction":
                    # Call the actual function with the provided arguments
                    try:
                        predictions = self.get_nutrition_prediction(**function_call.args)
                        
                        # Get explanation for the predictions
                        explanation_prompt = f"""Here are the nutrition predictions:
                        {json.dumps(predictions, indent=2)}
                        
                        Please explain these recommendations in a farmer-friendly way."""
                        
                        explanation_response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=explanation_prompt)]
                            )]
                        )
                        
                        response_text = explanation_response.text
                        
                    except ValueError as e:
                        response_text = f"I apologize, but there was an error with the values: {str(e)}"
                else:
                    response_text = response.text
            else:
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
        print("\nCattle Nutrition Advisor ready! Let's help you determine the proper nutrition for your cattle.")
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
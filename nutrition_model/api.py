from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from cattle_advisor import CattleNutritionAdvisor

# Load environment variables
load_dotenv()

app = FastAPI(title="Cattle Nutrition Advisor API")

INTERNAL_TO_API_MAPPING = {
    "DM Intake (lbs/day)": "dmIntakePerDay",
    "TDN (% DM)": "tdnPercentDm",
    "NEm (Mcal/lb)": "nemMcalPerLb",
    "NEg (Mcal/lb)": "negMcalPerLb",
    "CP (% DM)": "cpPercentDm",
    "Ca (%DM)": "caPercentDm",
    "P (% DM)": "pPercentDm",
    "TDN (lbs)": "tdnLbs",
    "NEm (Mcal)": "nemMcal",
    "NEg (Mcal)": "negMcal",
    "CP (lbs)": "cpLbs",
    "Ca (grams)": "caGrams",
    "P (grams)": "pGrams"
}

API_TO_INTERNAL_MAPPING = {v: k for k, v in INTERNAL_TO_API_MAPPING.items()}

def translate_to_api_format(predictions: Dict) -> Dict:
    """Translate predictions to API format (camelCase keys)"""
    return {INTERNAL_TO_API_MAPPING[key]: value for key, value in predictions.items()}

def translate_to_internal_format(predictions: Dict) -> Dict:
    """Translate API format back to internal format"""
    return {API_TO_INTERNAL_MAPPING[key]: value for key, value in predictions.items()}

# Get API key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")

# Initialize the advisor
advisor = CattleNutritionAdvisor(api_key=api_key)

class NutritionRequest(BaseModel):
    type_val: int  # 0, 1, or 2
    target_weight: float
    body_weight: float
    adg: float

class FeedRecommendationRequest(BaseModel):
    predictions: dict
    unavailable_ingredients: Optional[List[str]] = None

@app.post("/predict/nutrition")
async def predict_nutrition(request: NutritionRequest):
    try:
        # Map type value back to string
        type_mapping_reverse = {v: k for k, v in advisor.type_mapping.items()}
        cattle_type = type_mapping_reverse.get(request.type_val)
        
        if cattle_type is None:
            raise HTTPException(status_code=400, detail=f"Invalid type value. Must be one of: 0, 1, 2")
        
        predictions = advisor.get_nutrition_prediction(
            cattle_type=cattle_type,
            target_weight=request.target_weight,
            body_weight=request.body_weight,
            adg=request.adg
        )
        
        # Convert to API format (camelCase keys)
        api_format_predictions = translate_to_api_format(predictions)
        return api_format_predictions
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate/feed-recommendation")
async def generate_feed_recommendation(request: FeedRecommendationRequest):
    try:
        if request.unavailable_ingredients is not None and len(request.unavailable_ingredients) == 0:
            request.unavailable_ingredients = None
            
        # Convert API format back to internal format
        internal_predictions = translate_to_internal_format(request.predictions)
        
        recommendation = advisor.generate_feed_recommendation(
            predictions=internal_predictions,
            unavailable_ingredients=request.unavailable_ingredients
        )
        
        return {"recommendation": recommendation}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

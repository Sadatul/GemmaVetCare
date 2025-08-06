from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from cattle_advisor import CattleNutritionAdvisor

# Load environment variables
load_dotenv()

app = FastAPI(title="Cattle Nutrition Advisor API")

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
        
        return predictions
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate/feed-recommendation")
async def generate_feed_recommendation(request: FeedRecommendationRequest):
    try:
        if request.unavailable_ingredients is not None and len(request.unavailable_ingredients) == 0:
            request.unavailable_ingredients = None
        recommendation = advisor.generate_feed_recommendation(
            predictions=request.predictions,
            unavailable_ingredients=request.unavailable_ingredients
        )
        
        return {"recommendation": recommendation}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

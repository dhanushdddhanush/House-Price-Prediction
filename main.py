from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
# Load the trained model
model = joblib.load("house_price_model.pkl")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class HouseData(BaseModel):
    size_sqft: float

@app.get("/")
async def get_users():
    return {"message": "House Price Preediction Api!"}



# Define a prediction endpoint
@app.post("/predict/")
def predict_price(data: HouseData):
    input_data = [[data.size_sqft]]
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}





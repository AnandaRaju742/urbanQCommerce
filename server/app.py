import uvicorn
from openenv.core.env_server.http_server import create_app

# Bulletproof Docker Imports
try:
    from ..models import UrbanQCommerceAction, UrbanQCommerceObservation
except ImportError:
    from models import UrbanQCommerceAction, UrbanQCommerceObservation

from .urban_q_commerce_environment import UrbanQCommerceEnvironment

# 1. Define the App
app = create_app(
    UrbanQCommerceEnvironment,
    UrbanQCommerceAction,
    UrbanQCommerceObservation,
    env_name="urban_q_commerce",
    max_concurrent_envs=1,
)

@app.get("/")
def root_healthcheck():
    return {"status": "ok", "message": "Hugging Face Health Check Passed!"}

# 2. Add the Mandatory main() function for the Validator
def main():
    """Entry point for multi-mode deployment validation."""
    print("🚀 Starting Urban Q-Commerce Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 3. Add the Mandatory __name__ check
if __name__ == "__main__":
    main()
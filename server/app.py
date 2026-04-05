from openenv.core.env_server.http_server import create_app

# THE FIX: Bulletproof Docker Imports
try:
    from ..models import UrbanQCommerceAction, UrbanQCommerceObservation
except ImportError:
    from models import UrbanQCommerceAction, UrbanQCommerceObservation

from .urban_q_commerce_environment import UrbanQCommerceEnvironment

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
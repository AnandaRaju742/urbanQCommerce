from openenv.core.env_server.http_server import create_app

# Support both in-repo and standalone imports just like the official template
try:
    from ..models import UrbanQCommerceAction, UrbanQCommerceObservation
    from .environment import UrbanQCommerceEnvironment
except ImportError:
    from urban_q_commerce.models import UrbanQCommerceAction, UrbanQCommerceObservation
    from urban_q_commerce.server.environment import UrbanQCommerceEnvironment

# Wire up OUR custom environment and actions to the framework
app = create_app(
    UrbanQCommerceEnvironment, 
    UrbanQCommerceAction, 
    UrbanQCommerceObservation, 
    env_name="urban_q_commerce",
    max_concurrent_envs=10
)

def main():
    import uvicorn
    # Hardcoded to 8000, matching what we put in the README app_port!
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
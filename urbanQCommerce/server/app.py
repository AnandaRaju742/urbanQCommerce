from openenv.core.env_server.http_server import create_app
try:
    from ..models import UrbanQCommerceAction, UrbanQCommerceObservation
    from .urban_q_commerce_environment import UrbanQCommerceEnvironment
except (ImportError, ValueError):
    from models import UrbanQCommerceAction, UrbanQCommerceObservation
    from server.urban_q_commerce_environment import UrbanQCommerceEnvironment

app = create_app(
    UrbanQCommerceEnvironment,
    UrbanQCommerceAction,
    UrbanQCommerceObservation,
    env_name='urban_q_commerce',
    max_concurrent_envs=1,
)

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'message': 'Urban Q-Commerce is live!'}
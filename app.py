from src.app_factory import create_app
from src.core.config.config import Config


if __name__ == "__main__":
    print("Running")
    app = create_app()
    config = Config().dev_config
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
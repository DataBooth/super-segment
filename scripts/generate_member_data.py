import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from faker import Faker
import logfire
import os
from dotenv import load_dotenv

load_dotenv()

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    #print(OmegaConf.to_yaml(cfg))
    logfire.info("Config used", config=OmegaConf.to_container(cfg))
    fake = Faker()
    logfire.info("Generating member data", n=cfg.n_members)
    data = []
    for _ in range(cfg.n_members):
        data.append({
            "member_id": fake.uuid4(),
            "name": fake.name(),
            "email": fake.email(),
            "last_login_days": fake.random_int(min=1, max=180),
            "region": fake.state_abbr(),
            "balance": fake.pyfloat(left_digits=5, right_digits=2, positive=True),
        })
    df = pd.DataFrame(data)
    df.to_parquet(cfg.output_file)
    logfire.info("Saved members.parquet", path=cfg.output_file)

if __name__ == "__main__":
    main()

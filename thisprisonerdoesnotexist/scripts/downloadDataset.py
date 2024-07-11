from git import Repo

from thisprisonerdoesnotexist.src.trainingConfig import TrainingConfig
from thisprisonerdoesnotexist.src.unzip import unzip_file

print("cloning repo")
Repo.clone_from(TrainingConfig.dataset_url, TrainingConfig.download_data_dir)
print("done")
print("unzipping files")
unzip_file(
    f"{TrainingConfig.download_data_dir}/front.zip", TrainingConfig.download_data_dir
)
print("done")

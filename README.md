# ThisPrisonerDoesNotExist
<h2>
About the Project
</h2>
This project aims to create an AI model capable of generating realistic prisoners' mugshots. The model uses advanced machine learning techniques. To achieve this, we decided to develop a diffusion model on the Unet architecture
<h2>
  Dataset
</h2>


To train models we used illinois doc labelled faces dataset found on kaggle - https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset

However, we first performed a data cleaning


<img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/5add5742-3847-490c-9421-96c4bf0a8425" width="125" height="125"> <img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/4d630bbe-cc51-4dc3-8709-b30c6d763a6d" width="125" height="125">

In the process of data cleaning we found 1357 
prisoners with missing photos and one with swapped front and side photos

We hosted cleaned dataset on hiuggingface hub for easier access - https://huggingface.co/datasets/MGKK/Prisonersi

<h2>
  Data distribution
</h2>

![image](https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/a59a5b83-16aa-417d-9ef4-aaad424a53f2)

<h2>
  Results
</h2>
<p>We managed to train 2 models</p>
<p> PrisonersGeneratorx64 - model that generates mugshots in resolution 64x64 - https://huggingface.co/MGKK/PrisonersGeneratorx64 </p>
<p>PrisonersGeneratorx128 model that generates mugshots in resolution 128x128 - https://huggingface.co/MGKK/PrisonersGeneratorx128 </p>

<h3>Photo samples showing the progression of model learning on the same noises</h3>  

<p>PrisonersGeneratorx64:</p>
<img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/f4bbde3f-717d-41fd-a847-2a4ac4054eba" width="250" height="250"> <img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/729ac606-bbdc-45c9-86a9-e8182575fc79" width="250" height="250"> <img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/2db746b2-b846-406b-9fd7-d0d625fcf9ff" width="250" height="250">
<p>PrisonersGeneratorx128:</p>
<img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/c44fa86a-481f-467a-8dbf-e1007b789a83" width="250" height="250"> <img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/1846560c-594c-4fb5-95d2-58b4c8f2cffe" width="250" height="250"> <img src="https://github.com/ThisPrisonerDoesNotExist/ThisPrisonerDoesNotExist/assets/88160078/5c084fc0-d55b-45c0-9c15-2b2f477c722c" width="250" height="250">


Link to example notebook that shows how model works - https://colab.research.google.com/drive/1K0Z79b7U6lcfm12fuJ0VMVv1ks0yD008?authuser=1#scrollTo=_wnKZuaAk78D

same notebook you can find in "notebook" directory

Link to model that generates images in resolution 64x64 - https://huggingface.co/MGKK/PrisonersGeneratorx64

Link to model that generates images in resolution 128x128 - https://huggingface.co/MGKK/PrisonersGeneratorx128

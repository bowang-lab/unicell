from setuptools import setup, find_namespace_packages

with open("README.md", "r") as f:
  long_description = f.read()

setup(name="UniCell",
			packages=find_namespace_packages(include=["optimizers", "dataloaders","models"]),
      version="0.0.1",
      description="Universal cell segmentation",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="",
      author="Jun Ma",
      author_email="1259389904@qq.com",
      license="MIT",   
      platforms=["all"],
      install_requires=[      
            "monai",
						"numpy",
            "imagecodecs",
            "scipy",
						"scikit-image",
						"pillow",
						"tensorboard",
						"gdown",
						"torchvision",
						"tqdm",
						"psutil",
						"pandas",
						"einops",
            "numba",
            "matplotlib"
      ],
      entry_points={
          'console_scripts': [
              'unicell_train = unicell_train:main',
              'unicell_predict = unicell_predict:main',
              'com_metric = compute_metric:main',
          ],
      },
      )
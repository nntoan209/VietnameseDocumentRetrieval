from setuptools import setup, find_packages, Command

setup(
    name='LawRetrieval',
    version='1.2.8',
    description='LawRetrieval',
    author_email='nntoan209@gmail.com',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'prettytable',
        'transformers>=4.33.0',
        'datasets',
        'accelerate>=0.25.0',
        'huggingface_hub=0.20.1',
        'sentence_transformers',
        'underthesea',
        'rank_bm25',
        'sentencepiece',
        'deepspeed',
        'jsonlines',
        'rank-bm25',        
    ]
)

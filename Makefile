install:
	pyenv local chessenv
	pip install -r requirements.txt

pyenv:
	pyenv virtualenv 3.8.12 chessenv

install:
	pyenv local chessenv
	@pip install -e .

pyenv:
	pyenv virtualenv 3.8.12 chessenv

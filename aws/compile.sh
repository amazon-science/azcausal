cd ..

python setup.py sdist

cp dist/azcausal-*.tar.gz aws

mv aws/azcausal-*.tar.gz aws/azcausal.tar.gz
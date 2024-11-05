python experiment.py --evalfile finalidx100000.csv --batchsize 10000

## Changing prompt length
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 10
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 20
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 30
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 40
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 60
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 70
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 80
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 90
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 100
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 200
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 300
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 400
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 1000

## Changing completion length
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 10 --maxtokens 10
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 20 --maxtokens 20
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 30 --maxtokens 30
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 40 --maxtokens 40
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 60 --maxtokens 60
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 70 --maxtokens 70
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 80 --maxtokens 80
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 90 --maxtokens 90
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 100 --maxtokens 100
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 200 --maxtokens 200
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 300 --maxtokens 300
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 400 --maxtokens 400
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 500 --maxtokens 500
## python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --complen 1000 --maxtokens 1000

## Changing Prompt type
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype skipalt
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype end
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype corner
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --promptlen 500 --prompttype cornerdel


## Changing Temperature
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.1
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.2
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.3
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.4
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.5
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.6
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.7
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.8
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 0.9
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --temperature 1.0

## Changing Beam Width
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 2
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 3
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 4
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --beamwidth 5

## Changing Time Step/Revision
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step100000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step105000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step110000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step115000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step120000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step125000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step130000
python experiment.py --evalfile finalidx100000.csv --batchsize 10000 --modelstep step135000
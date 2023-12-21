# $1 - algorithm
# $2 - instance
# $3 - timelimit
# $4 - N
# $5 - offset
if [ $# != 5 ]
then
  echo "$0: Missing arguments"
  exit 1
else
  mkdir "results/"$2"_result"
  : > "results/"$2
  for i in `seq 1 $4`
  do
    hostname
    w
    pypy main.py $1 $2 $3
    n=`expr $5 + $i`
    cp "results/"$2 "results/"$2"_result/"$1"_"$n
    : > "results/"$2
  done
fi
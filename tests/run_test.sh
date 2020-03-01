#!/bin/bash

TEST_DIFF=$1
LEFT=$2
RIGHT=$3
WPATH=$4
DIFFNAME=$5

echo "Check $2 $3"



${TEST_DIFF} -V -a 1e-6 -r 1e-8 -s ' \t\n:<>=,;' $LEFT $RIGHT >$WPATH/diff_output

ret=$?
if [[ "$ret" == "0" ]]
then
   exit 0
fi

cat $WPATH/diff_output | head

echo "Diff Output:"
echo ""
# reverse needed, so we can use it to patch
cd $WPATH
diff -u $RIGHT $LEFT >$WPATH/$DIFFNAME
echo ""

exit 1

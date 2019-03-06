#!/bin/sh

wget https://storage.googleapis.com/logicai/recsys2019/trivagoRecSysChallengeData2019.zip
mv trivagoRecSysChallengeData2019/*.csv .
rm -rf trivagoRecSysChallengeData2019
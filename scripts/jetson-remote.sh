#!/bin/bash
printf "==============================================================\n"
printf "The login is seniordesign and the password is seniordesign\n"
printf "==============================================================\n"
sleep 2
ssh -L 4002:localhost:4000 seniordesign@seniordesign.local

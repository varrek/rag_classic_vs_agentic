#!/bin/bash
# Script to increase inotify watch limit

# Display current limits
echo "Current inotify limits:"
cat /proc/sys/fs/inotify/max_user_watches
cat /proc/sys/fs/inotify/max_user_instances

# Increase the limit temporarily (until next reboot)
echo "Increasing inotify watch limit temporarily..."
sudo sh -c 'echo 65536 > /proc/sys/fs/inotify/max_user_watches'
sudo sh -c 'echo 256 > /proc/sys/fs/inotify/max_user_instances'

# Make the change permanent
echo "Making the change permanent..."
sudo sh -c 'echo "fs.inotify.max_user_watches = 65536" >> /etc/sysctl.conf'
sudo sh -c 'echo "fs.inotify.max_user_instances = 256" >> /etc/sysctl.conf'
sudo sysctl -p

# Display new limits
echo "New inotify limits:"
cat /proc/sys/fs/inotify/max_user_watches
cat /proc/sys/fs/inotify/max_user_instances

echo "inotify watch limit increased successfully." 

# Full ex7 pipeline:
# The full pipeline will be:
#
# 1. Build the original pose graph from Exercise 6.
# 2. For each current keyframe c_n, consider older keyframes c_i.
# 3. Estimate how uncertain the relative pose c_i \rightarrow c_n is using the graph.
# 4. Keep plausible candidates.
# 5. Try visual matching and PnP/RANSAC for those candidates.
# 6. Add a BetweenFactorPose3 for accepted loops.
# 7. Re-optimize the pose graph.
# 8. Show that the trajectory and its uncertainty improve.

# You are not searching for the shortest visual route between two frames.
# You are searching for the least uncertain already-known geometric route
# through the current pose graph.
def main():
    # Question 1 - Relative Covariance
    pass

if __name__ == '__main__':
    main()
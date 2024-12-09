using AMP;
using UnityEngine;
using System.Collections.Generic;

public abstract class TrainingEnv : MonoBehaviour
{
    public abstract void BeginEpisode();
    public abstract float GetReward(JointDriveController controller);

    public abstract List<float> GetGoals(Skeleton skeleton);

    public abstract bool ValidateEnvironment(Skeleton skeleton);
}

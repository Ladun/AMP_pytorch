using AMP;
using UnityEngine;

public abstract class TrainingEnv : MonoBehaviour
{
    public abstract void BeginEpisode();
    public abstract float GetReward(JointDriveController jdController);
}

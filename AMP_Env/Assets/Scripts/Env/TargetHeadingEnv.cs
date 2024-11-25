using UnityEngine;
using AMP;
using System.Collections.Generic;

public class TargetHeadingEnv : TrainingEnv
{

    private float targetSpeed = 1;
    private float targetHeading = 0;

    public override void BeginEpisode()
    {
        targetSpeed = Random.Range(1f, 5f);
        targetHeading = Random.Range(0, 1f) * Mathf.PI * 2;
    }

    public override List<float> GetGoals(Skeleton skeleton)
    {
        float tarHeading = targetHeading;
        float tarSpeed = targetSpeed;

        Transform root = skeleton.GetJoints()[0];
        Vector3 d = root.rotation * Vector3.right;
        float characterHeading = Mathf.Atan2(-d.z, d.x);
        tarHeading -= characterHeading;

        return new List<float> { Mathf.Cos(tarHeading), -Mathf.Sin(tarHeading), tarSpeed };
    }

    public override float GetReward(JointDriveController jdController)
    {

        Vector3 cmv = Vector3.zero;
        float totalMass = 0;
        foreach(var bp in jdController.bodyPartsList)
        {
            cmv += bp.rb.linearVelocity * bp.rb.mass;
            totalMass += bp.rb.mass;
        }
        cmv = cmv / totalMass;
        cmv.y = 0;

        Vector3 targetDir = new Vector3(Mathf.Cos(targetHeading), 0, -Mathf.Sin(targetHeading));

        return Mathf.Exp(-0.25f * (targetSpeed - Vector3.Dot(targetDir, cmv)));
    }
}

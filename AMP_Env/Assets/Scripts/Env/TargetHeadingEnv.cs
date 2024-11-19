using UnityEngine;
using AMP;

public class TargetHeadingEnv : TrainingEnv
{

    private float targetSpeed = 1;
    private float targetHeading = 0;

    public override void BeginEpisode()
    {
        targetSpeed = Random.Range(1f, 5f);
        targetHeading = Random.Range(0, 1f) * Mathf.PI * 2;
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

using UnityEngine;
using AMP;
using System.Collections.Generic;

public class TargetHeadingEnv : TrainingEnv
{
    public Direction direction;

    public bool usingArticulationPenalty = false;
    public float velRewardScale = 1.0f;

    private float targetSpeed = 1;
    private float targetHeading = 0;

    public Vector3 center;
    public Vector3 size;

    public override void BeginEpisode()
    {
        targetSpeed = Random.Range(1f, 5f);
        targetHeading = Random.Range(0, 1f) * Mathf.PI * 2;


        if (direction != null)
        {
            direction.SetHeading(targetHeading);
        }
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

    public override float GetReward(ArticulationBodyController controller)
    {

        Vector3 cmv = Vector3.zero;
        float totalMass = 0;
        foreach(var bp in controller.bodyPartsList)
        {
            cmv += bp.ab.linearVelocity * bp.ab.mass;
            totalMass += bp.ab.mass;
        }
        cmv = cmv / totalMass;
        cmv.y = 0;
        cmv.z *= -1;

        Vector3 targetDir = new Vector3(Mathf.Cos(targetHeading), 0, -Mathf.Sin(targetHeading));
        float avg_speed = Vector3.Dot(targetDir, cmv);

        float vel_reward = 0;
        if (avg_speed > 0)
        {
            float vel_err = targetSpeed - avg_speed;
            vel_reward = Mathf.Exp(-velRewardScale * vel_err * vel_err); 
        }

        if(usingArticulationPenalty)
        {
            float penalty = 0;
            float jointCnt = 0;
            foreach (var bp in controller.bodyPartsList)
            {
                if(bp.IsPenalizable())
                {
                    penalty += bp.GetBoundsPenalty();
                    jointCnt++;
                }
            }
            if (jointCnt > 0)
                penalty /= jointCnt;
            vel_reward += Mathf.Log(1 - penalty);
        }

        return vel_reward;
    }

    public override bool ValidateEnvironment(Skeleton skeleton)
    {
        Bounds b = new Bounds(transform.position + center, size);

        return b.Contains(skeleton.transform.position);
    }


    private void OnDrawGizmosSelected()
    {
        Bounds b = new Bounds(transform.position + center, size);
        Gizmos.DrawWireCube(b.center, b.size);
    }
}

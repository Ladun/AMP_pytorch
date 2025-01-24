using AMP;
using System.Collections.Generic;
using System.Drawing;
using UnityEngine;

public class TargetLocationEnv : TrainingEnv
{
    private Vector3 targetLocation;
    private float targetSpeed = 1;

    public Vector3 center;
    public Vector3 size;

    public override void BeginEpisode()
    {
        targetSpeed = Random.Range(1f, 2f);

        // TODO: randomized spawn target location

        float rad = Random.Range(0, 360) * Mathf.Deg2Rad;
        float radius = Random.Range(5, size.x - 10);
        targetLocation = new Vector3(Mathf.Cos(rad) * radius, 1, Mathf.Sin(rad) * radius);
    }

    public override List<float> GetGoals(Skeleton skeleton)
    {
        Transform root = skeleton.GetRoot();

        var tp = new Vector3(targetLocation.x, root.position.y, targetLocation.z);
        var localizedTarget = root.InverseTransformPoint(tp);

        return new List<float> { localizedTarget.x, localizedTarget.y, localizedTarget.z };
    }

    public override float GetReward(ArticulationBodyController controller)
    {

        Vector3 cmv = Vector3.zero;
        float totalMass = 0;
        foreach (var bp in controller.bodyPartsList)
        {
            cmv += bp.ab.linearVelocity * bp.ab.mass;
            totalMass += bp.ab.mass;
        }
        cmv = cmv / totalMass;
        cmv.y = 0;

        Transform root = controller.bodyPartsDict[0].ab.transform;
        var tp = new Vector3(targetLocation.x, 0, targetLocation.z);
        var rp = new Vector3(root.position.x, 0, root.position.z);
        Vector3 targetDir = (tp - rp).normalized;

        float f = 0.7f * Mathf.Exp(-0.5f * (targetLocation - root.position).sqrMagnitude);
        float s = 0.3f * Mathf.Exp(-(Mathf.Max(0, targetSpeed - Vector3.Dot(targetDir, cmv))));
        
        return f + s;
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

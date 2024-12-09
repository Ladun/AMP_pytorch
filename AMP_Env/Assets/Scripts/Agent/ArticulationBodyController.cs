using System.Collections.Generic;
using System.Data;
using Unity.MLAgents;
using UnityEngine;
using UnityEngine.Serialization;
using static UnityEngine.GraphicsBuffer;


namespace AMP
{
    [System.Serializable]
    public class ArticulationBodyPart
    {
        [Header("Body Part Info")]
        public ArticulationBody ab;

        public Vector3 lowerLim;
        public Vector3 upperLim;

        [Header("Ground & Target Contact")]
        [Space(10)]
        public GroundContact groundContact;

        public TargetContact targetContact;

        [FormerlySerializedAs("thisABController")]
        [HideInInspector] public ArticulationBodyController thisABController;

        public void SetJointTargetFromRotVector(List<float> f)
        {
            if (ab.jointType == ArticulationJointType.SphericalJoint)
            {
                float x = f[0];
                float y = f[1];
                float z = f[2];

                float theta = Mathf.Sqrt(x * x + y * y + z * z);
                if (theta != 0)
                {
                    x = (x / theta);
                    y = (y / theta);
                    z = (z / theta);
                }

                Quaternion rot = new Quaternion(x * Mathf.Sign(theta / 2),
                                                y * Mathf.Sign(theta / 2),
                                                z * Mathf.Sign(theta / 2),
                                                Mathf.Cos(theta / 2));
                Vector3 euler = rot.eulerAngles;

                euler.x = Mathf.Clamp(euler.x, ab.xDrive.lowerLimit, ab.xDrive.upperLimit);
                ab.SetDriveTarget(ArticulationDriveAxis.X, euler.x);
                euler.y = Mathf.Clamp(euler.y, ab.yDrive.lowerLimit, ab.yDrive.upperLimit);
                ab.SetDriveTarget(ArticulationDriveAxis.Y, euler.y);
                euler.z = Mathf.Clamp(euler.z, ab.zDrive.lowerLimit, ab.zDrive.upperLimit);
                ab.SetDriveTarget(ArticulationDriveAxis.Z, euler.z);
            }

            else if(ab.jointType == ArticulationJointType.RevoluteJoint)
            {
                float q = f[0]* Mathf.Rad2Deg;
                q = Mathf.Clamp(q, ab.xDrive.lowerLimit, ab.xDrive.upperLimit);
                ab.SetDriveTarget(ArticulationDriveAxis.X, q);
            }

        }
    }

    public class ArticulationBodyController : MonoBehaviour
    {
        [Header("Drive info")]
        public float stiffness = 100f; 
        public float damping = 10f;    


        [HideInInspector] public Dictionary<int, ArticulationBodyPart> bodyPartsDict = new Dictionary<int, ArticulationBodyPart>();
        [HideInInspector] public List<ArticulationBodyPart> bodyPartsList = new List<ArticulationBodyPart>();

        public void SetupBodyPart(int key, Transform t)
        {
            ArticulationBody ab = t.GetComponent<ArticulationBody>();
            var bp = new ArticulationBodyPart
            {
                ab = ab
            };

            bp.groundContact = t.GetComponent<GroundContact>();
            if (!bp.groundContact)
            {
                bp.groundContact = t.gameObject.AddComponent<GroundContact>();
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }
            else
            {
                bp.groundContact.agent = gameObject.GetComponent<Agent>();
            }

            bp.thisABController = this;
            bodyPartsDict.Add(key, bp);
            bodyPartsList.Add(bp);
        }

        public void ResetState()
        {
            bodyPartsDict.Clear();
            bodyPartsList.Clear();
        }
    }

}
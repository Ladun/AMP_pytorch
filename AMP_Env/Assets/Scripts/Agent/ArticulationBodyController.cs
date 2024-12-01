using System.Collections.Generic;
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

        [Header("Ground & Target Contact")]
        [Space(10)]
        public GroundContact groundContact;

        public TargetContact targetContact;

        [FormerlySerializedAs("thisABController")]
        [HideInInspector] public ArticulationBodyController thisABController;

        public void SetJointTargetFromRotVector(float x, float y, float z)
        {
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
                ab = ab,
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


            ArticulationDrive xDrive = ab.xDrive;
            xDrive.stiffness = stiffness;
            xDrive.damping = damping;
            ab.xDrive = xDrive;

            if(ab.jointType == ArticulationJointType.SphericalJoint)
            {

                ArticulationDrive yDrive = ab.xDrive;
                yDrive.stiffness = stiffness;
                yDrive.damping = damping;
                ab.yDrive = yDrive;
                ArticulationDrive zDrive = ab.xDrive;
                zDrive.stiffness = stiffness;
                zDrive.damping = damping;
                ab.xDrive = zDrive;
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
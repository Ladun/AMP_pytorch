using UnityEngine;

public class Direction : MonoBehaviour
{
    public Transform p;

    public Transform target;
    public float heading;

    // Update is called once per frame
    void Update()
    {
        if (p != null)
        {
            Vector3 parentPos = p.childCount > 0 ? p.GetChild(0).position : p.position;
            parentPos.y = transform.position.y;
            transform.position = parentPos;
        }

        if(target != null)
        {
            Vector3 targetPos = target.position;
            targetPos.y = transform.position.y;


        }
    }

    public void SetHeading(float heading)
    {
        this.heading = heading;
        transform.eulerAngles = Vector3.up * heading * Mathf.Rad2Deg;
    }

    private void OnDrawGizmosSelected()
    {
        float h = heading * Mathf.Deg2Rad;
        Vector3 dir = new Vector3(Mathf.Cos(h), 0, -Mathf.Sin(h));
        Debug.DrawRay(transform.position, dir);

        Vector3 d = transform.rotation * Vector3.right;
        float characterHeading = Mathf.Atan2(-d.z, d.x);
        Debug.DrawRay(transform.position, d, Color.blue);

        float tarHeading = h - characterHeading;
        Debug.DrawRay(transform.position, new Vector3(Mathf.Cos(tarHeading), 0, -Mathf.Sin(tarHeading)), Color.red);
    }
}

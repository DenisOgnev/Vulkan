#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

const float YAW         =  180.0f;
const float PITCH       =  0.0f;
const float SPEED       =  2.5f;
const float SENSITIVITY =  0.1f;
const float ZOOM        =  45.0f;


class Camera
{
public:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;
    
    float yaw;
    float pitch;
    
    float movement_speed;
    float mouse_sensitivity;
    float zoom;

    Camera(glm::vec3 position_vector = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up_vector = glm::vec3(0.0f, 0.0f, 1.0f), float yaw_value = YAW, float pitch_value = PITCH) : front(glm::vec3(-1.0f, 0.0f, 0.0f)), movement_speed(SPEED), mouse_sensitivity(SENSITIVITY), zoom(ZOOM)
    {
        position = position_vector;
        world_up = up_vector;
        yaw = yaw_value;
        pitch = pitch_value;

        update_camera_vectors();
    }

    glm::mat4 get_view_matrix()
    {
        return glm::lookAt(position, position + front, up);
    }

    void process_keyboard_input(Camera_Movement direction, float delta_time)
    {
        float velocity = movement_speed * delta_time;
        if (direction == FORWARD)
            position += front * velocity;
        if (direction == BACKWARD)
            position -= front * velocity;
        if (direction == LEFT)
            position -= right * velocity;
        if (direction == RIGHT)
            position += right * velocity;
    }

    void process_mouse_input(float x_offset, float y_offset)
    {
        x_offset *= mouse_sensitivity;
        y_offset *= mouse_sensitivity;

        yaw -= x_offset;
        pitch += y_offset;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        update_camera_vectors();
    }

    void process_mouse_scroll(float yoffset)
    {
        zoom -= (float)yoffset;
        if (zoom < 1.0f)
            zoom = 1.0f;
        if (zoom > 45.0f)
            zoom = 45.0f; 
    }

private:
    void update_camera_vectors()
    {
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.z = sin(glm::radians(pitch));

        front = glm::normalize(front);
        
        right = glm::normalize(glm::cross(front, world_up));  
        up = glm::normalize(glm::cross(right, front));
    }
};

#endif
#ifndef EVENTGROUP_H
#define EVENTGROUP_H

class EventGroup
{
public:
    typedef unsigned char EventGroupHandle_t;
    typedef unsigned char EventBits_t;

    EventGroup();

    void setBits(EventGroupHandle_t handle, EventBits_t bit);
    void clearBits(EventGroupHandle_t handle, EventBits_t bit);
    bool waitBits(EventGroupHandle_t handle, EventBits_t bit);

    void setBits(EventBits_t bit);
    void clearBits(EventBits_t bit);
    bool waitBits(EventBits_t bit);

private:
    EventGroupHandle_t handle;

};

#endif // EVENTGROUP_H

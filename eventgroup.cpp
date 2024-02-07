#include "eventgroup.h"

EventGroup::EventGroup()
{
    this->handle = 0;
}


void EventGroup::setBits(EventGroupHandle_t handle, EventBits_t bit)
{
    handle = handle | (1 << bit);
}

void EventGroup::clearBits(EventGroupHandle_t handle, EventBits_t bit)
{
    handle = handle & (~(1 << bit));
}

bool EventGroup::waitBits(EventGroupHandle_t handle, EventBits_t bit)
{
    if (handle & (1 << bit)) {
        return true;
    } else {
        return false;
    }
}



void EventGroup::setBits(EventBits_t bit)
{
    this->handle = this->handle | (1 << bit);
}

void EventGroup::clearBits(EventBits_t bit)
{
    this->handle = this->handle & (~(1 << bit));
}

bool EventGroup::waitBits(EventBits_t bit)
{
    if (this->handle & (1 << bit)) {
        return true;
    } else {
        return false;
    }
}

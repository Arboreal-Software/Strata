#include "owned_admission_cancel.hpp"
#include <cstdlib>
static void require(bool ok){if(!ok)std::abort();}
using strata::program::OwnedAdmissionCancel;
int main(){
 OwnedAdmissionCancel q;int slot=-1;uint64_t key=0;bool active=false,cancelled=false;
 require(OwnedAdmissionCancel::parse("BGEN 0 12 http_request=012345abcdef 1,2,3",false,slot,key));
 require(slot==0&&q.enqueue(slot,key));
 // Cancellation arrives before the main reactor begins its queued request.
 require(q.cancel(slot,key,active)&&!active);
 require(q.begin(slot,key,cancelled)&&cancelled);
 uint64_t other=key+1;require(q.enqueue(1,other));
 require(!q.cancel(1,key,active)); // wrong request cannot cancel the other lane
 require(q.begin(1,other,cancelled)&&!cancelled);
 require(q.cancel(1,other,active)&&active);
 q.finish(0,key);require(q.enqueue(0,other+1));
 require(!q.cancel(0,key,active)); // stale cleanup cannot affect reused admission
 q.finish(0,key);require(q.begin(0,other+1,cancelled)&&!cancelled);
 require(!OwnedAdmissionCancel::parse("BCANCEL 0 request=012345abcdef junk",true,slot,key));
 require(!OwnedAdmissionCancel::parse("BGEN 0 1 http_request=012345abcdef http_request=012345abcdee 1",false,slot,key));
 require(!OwnedAdmissionCancel::parse("BGEN 2 1 http_request=012345abcdef 1",false,slot,key));
 require(OwnedAdmissionCancel::parse("BCANCEL 1 request=000000000000",true,slot,key)&&key==1);
 require(!OwnedAdmissionCancel::parse("BCANCEL 1 request=012345ABCDEf",true,slot,key));
 // Private preparation/capture/reentry preserves ONE original rank/incarnation.
 OwnedAdmissionCancel images;require(OwnedAdmissionCancel::parse("BIPREP 0 32768 3 20 0 http_request=012345abcdef",false,slot,key));
 require(images.prepare_image(slot,key));const auto first=images.incarnation(slot,key);
 require(first&&images.image_current(slot,key,first,false));
 require(!images.enqueue(slot,key)&&!images.prepare_image(slot,key));
 require(!images.cancel(slot,key,active)); // public text cancellation cannot borrow private image identity
 require(!images.cancel_image(slot,key,first+1,active));
 require(images.deliver_image(slot,key)&&!images.deliver_image(slot,key));
 require(images.incarnation(slot,key)==first&&images.image_current(slot,key,first,true));
 require(images.begin(slot,key,cancelled)&&!cancelled);
 require(images.cancel_image(slot,key,first,active)&&active);
 images.finish(slot,key);require(images.prepare_image(slot,key));
 require(images.incarnation(slot,key)>first&&!images.cancel_image(slot,key,first,active));
 uint64_t incarnation=0;bool cancel=false;
 require(OwnedAdmissionCancel::image_control("BICLOSE 0 request=012345abcdef incarnation=7 cancel=1","BICLOSE ",slot,key,incarnation,cancel)&&incarnation==7&&cancel);
 for(const auto* invalid:{"BICLOSE 0 request=012345abcdef incarnation=0 cancel=1",
     "BICLOSE 0 request=012345abcdef incarnation=+7 cancel=1",
     "BICLOSE 0 request=012345abcdef incarnation=7 cancel=1 trailing",
     "BICLOSE 2 request=012345abcdef incarnation=7 cancel=1"})
  require(!OwnedAdmissionCancel::image_control(invalid,"BICLOSE ",slot,key,incarnation,cancel));

}

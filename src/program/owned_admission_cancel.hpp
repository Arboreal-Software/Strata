#pragma once
#include <array>
#include <charconv>
#include <cstdint>
#include <string_view>
namespace strata::program {
// Fixed storage, serialized by StdinShared::mu. Only a queued BGEN creates an
// admission identity. This never confers authority over an occupied batch slot.
struct OwnedAdmissionCancel {
    struct Entry { uint64_t key=0; bool cancelled=false, active=false;
        uint64_t incarnation=0; bool image_preparing=false,image_delivered=false; };
    uint64_t next_incarnation=0;
    std::array<Entry,2> entries{};
    static bool key(std::string_view s,uint64_t& out) noexcept {
        if(s.size()!=12)return false;
        uint64_t v=0;for(char c:s){unsigned n;if(c>='0'&&c<='9')n=unsigned(c-'0');
            else if(c>='a'&&c<='f')n=unsigned(c-'a'+10);else return false;
            v=(v<<4)|n;}
        out=v+1;return true; // all-zero request keys remain representable
    }
    static bool parse(std::string_view line,bool cancel,int& slot,uint64_t& request) noexcept {
        const auto prefix=cancel?std::string_view("BCANCEL "):
            line.starts_with("BGENI ")?std::string_view("BGENI "):
            line.starts_with("BIPREP ")?std::string_view("BIPREP "):std::string_view("BGEN ");
        if(!line.starts_with(prefix))return false;
        line.remove_prefix(prefix.size());
        if(line.size()<3||line[0]<'0'||line[0]>'1'||line[1]!=' ')return false;
        slot=line[0]-'0';line.remove_prefix(2);
        if(cancel)return line.starts_with("request=")&&key(line.substr(8),request);
        bool found=false;
        while(!line.empty()){
            auto end=line.find(' ');auto word=line.substr(0,end);
            if(word.starts_with("http_request=")){
                if(found||!key(word.substr(13),request))return false;
                found=true;
            }
            if(end==line.npos)break;
            line.remove_prefix(end+1);
        }
        return found;
    }
    static bool image_control(std::string_view line,std::string_view prefix,int& slot,
        uint64_t& request,uint64_t& incarnation,bool& cancel) noexcept {
        if(!line.starts_with(prefix))return false;
        line.remove_prefix(prefix.size());
        if(line.size()<3||line[0]<'0'||line[0]>'1'||line[1]!=' ')return false;
        slot=line[0]-'0';line.remove_prefix(2);
        if(!line.starts_with("request=")||line.size()<21||!key(line.substr(8,12),request)||line[20]!=' ')return false;
        line.remove_prefix(21);if(!line.starts_with("incarnation="))return false;line.remove_prefix(12);
        const auto gap=line.find(' ');if(gap==line.npos)return false;
        const auto digits=line.substr(0,gap);const auto parsed=std::from_chars(digits.data(),digits.data()+digits.size(),incarnation);
        if(parsed.ec!=std::errc{}||parsed.ptr!=digits.data()+digits.size()||!incarnation)return false;
        line.remove_prefix(gap+1);if(line!="cancel=0"&&line!="cancel=1")return false;
        cancel=line=="cancel=1";return true;
    }
    bool enqueue(int slot,uint64_t request) noexcept {
        if(slot<0||slot>1||!request||entries[size_t(slot)].key||next_incarnation==UINT64_MAX)return false;
        entries[size_t(slot)]={request,false,false,++next_incarnation,false,false};return true;
    }
    // The producer's prepare and final delivery occupy ONE canonical entry.
    // Final BGENI cannot re-enqueue a request, refresh its rank, or resurrect it.
    bool prepare_image(int slot,uint64_t request) noexcept {
        if(!enqueue(slot,request))return false;
        entries[size_t(slot)].image_preparing=true;return true;
    }
    bool deliver_image(int slot,uint64_t request) noexcept {
        if(slot<0||slot>1)return false;
        auto& e=entries[size_t(slot)];
        if(e.key!=request||!e.image_preparing||e.image_delivered||e.cancelled||e.active)return false;
        e.image_delivered=true;return true;
    }
    uint64_t incarnation(int slot,uint64_t request)const noexcept {
        return slot>=0&&slot<2&&entries[size_t(slot)].key==request?entries[size_t(slot)].incarnation:0;
    }
    bool image_current(int slot,uint64_t request,uint64_t incarnation,bool delivered)const noexcept {
        if(slot<0||slot>1||!incarnation)return false;
        const auto& e=entries[size_t(slot)];
        return e.key==request&&e.incarnation==incarnation&&!e.cancelled&&e.image_preparing&&
            (!delivered||e.image_delivered);
    }
    bool cancel(int slot,uint64_t request,bool& active) noexcept {
        active=false;if(slot<0||slot>1||!request)return false;
        auto& e=entries[size_t(slot)];if(e.key!=request||e.image_preparing)return false;
        e.cancelled=true;active=e.active;return true;
    }
    bool cancel_image(int slot,uint64_t request,uint64_t incarnation,bool& active) noexcept {
        active=false;
        if(slot<0||slot>1||!incarnation)return false;
        auto& e=entries[size_t(slot)];
        if(e.key!=request||e.incarnation!=incarnation||!e.image_preparing)return false;
        e.cancelled=true;active=e.active;return true;
    }
    bool begin(int slot,uint64_t request,bool& cancelled) noexcept {
        if(slot<0||slot>1||!request)return false;
        auto& e=entries[size_t(slot)];if(e.key!=request||e.active||(e.image_preparing&&!e.image_delivered))return false;
        e.active=true;cancelled=e.cancelled;return true;
    }
    void finish(int slot,uint64_t request) noexcept {
        if(slot>=0&&slot<2&&entries[size_t(slot)].key==request)entries[size_t(slot)]={};
    }
};
}

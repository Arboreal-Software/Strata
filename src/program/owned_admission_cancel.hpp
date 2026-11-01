#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace strata::program {
// Fixed storage, serialized by StdinShared::mu. Only a queued BGEN creates an
// admission identity. This never confers authority over an occupied batch slot.
struct OwnedAdmissionCancel {
    struct Entry { uint64_t key=0; bool cancelled=false, active=false; };
    std::array<Entry,2> entries{};
    static bool key(std::string_view s,uint64_t& out) noexcept {
        if(s.size()!=12)return false;
        uint64_t v=0;for(char c:s){unsigned n;if(c>='0'&&c<='9')n=unsigned(c-'0');
            else if(c>='a'&&c<='f')n=unsigned(c-'a'+10);else return false;v=(v<<4)|n;}
        out=v+1;return true; // all-zero request keys remain representable
    }
    static bool parse(std::string_view line,bool cancel,int& slot,uint64_t& request) noexcept {
        const auto prefix=cancel?std::string_view("BCANCEL "):std::string_view("BGEN ");
        if(!line.starts_with(prefix))return false;line.remove_prefix(prefix.size());
        if(line.size()<3||line[0]<'0'||line[0]>'1'||line[1]!=' ')return false;
        slot=line[0]-'0';line.remove_prefix(2);
        if(cancel)return line.starts_with("request=")&&key(line.substr(8),request);
        bool found=false;
        while(!line.empty()){
            auto end=line.find(' ');auto word=line.substr(0,end);
            if(word.starts_with("http_request=")){
                if(found||!key(word.substr(13),request))return false;found=true;
            }
            if(end==line.npos)break;line.remove_prefix(end+1);
        }
        return found;
    }
    bool enqueue(int slot,uint64_t request) noexcept {
        if(slot<0||slot>1||!request||entries[size_t(slot)].key)return false;
        entries[size_t(slot)]={request,false,false};return true;
    }
    bool cancel(int slot,uint64_t request,bool& active) noexcept {
        active=false;if(slot<0||slot>1||!request)return false;
        auto& e=entries[size_t(slot)];if(e.key!=request)return false;
        e.cancelled=true;active=e.active;return true;
    }
    bool begin(int slot,uint64_t request,bool& cancelled) noexcept {
        if(slot<0||slot>1||!request)return false;
        auto& e=entries[size_t(slot)];if(e.key!=request||e.active)return false;
        e.active=true;cancelled=e.cancelled;return true;
    }
    void finish(int slot,uint64_t request) noexcept {
        if(slot>=0&&slot<2&&entries[size_t(slot)].key==request)entries[size_t(slot)]={};
    }
};
}

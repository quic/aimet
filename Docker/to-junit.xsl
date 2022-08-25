<?xml version="1.0" encoding="UTF-8"?>
<!--
The MIT License (MIT)

Copyright (c) 2014, Gregory Boissinot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-->
<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xunit="http://www.xunit.org">
    <xsl:output method="xml" indent="yes" encoding="UTF-8" cdata-section-elements="system-out system-err failure"/>
    <xsl:decimal-format decimal-separator="." grouping-separator=","/>

    <xsl:function name="xunit:junit-time" as="xs:string">
        <xsl:param name="value" as="xs:anyAtomicType?" />

        <xsl:variable name="time" as="xs:double">
            <xsl:choose>
                <xsl:when test="$value instance of xs:double">
                    <xsl:value-of select="$value" />
                </xsl:when>
                <xsl:otherwise>
                    <xsl:value-of select="translate(string(xunit:if-empty($value, 0)), ',', '.')" />
                </xsl:otherwise>
            </xsl:choose>
        </xsl:variable>
        <xsl:value-of select="format-number($time, '0.000')" />
    </xsl:function>

    <xsl:function name="xunit:if-empty" as="xs:string">
        <xsl:param name="value" as="xs:anyAtomicType?" />
        <xsl:param name="default" as="xs:anyAtomicType" />
        <xsl:value-of select="if (string($value) != '') then string($value) else $default" />
    </xsl:function>

    <xsl:function name="xunit:is-empty" as="xs:boolean">
        <xsl:param name="value" as="xs:string?" />
        <xsl:value-of select="string($value) != ''" />
    </xsl:function>

    <xsl:template match="/">
        <xsl:apply-templates/>
    </xsl:template>
    <xsl:template match="//testsuites">
        <testsuites>
            <xsl:apply-templates/>
        </testsuites>
    </xsl:template>
    <xsl:template match="//testsuite">
        <testsuite>
            <xsl:attribute name="name">
                <xsl:value-of select="@name"/>
            </xsl:attribute>
            <xsl:attribute name="tests">
                <xsl:value-of select="xunit:if-empty(@tests, 0)"/>
            </xsl:attribute>
            <xsl:attribute name="failures">
                <xsl:value-of select="xunit:if-empty(@failures, 0)"/>
            </xsl:attribute>
            <xsl:attribute name="errors">
                <xsl:value-of select="xunit:if-empty(@errors, 0)"/>
            </xsl:attribute>
            <xsl:attribute name="skipped">
                <xsl:value-of select="xunit:if-empty(@disabled, 0)"/>
            </xsl:attribute>
            <xsl:attribute name="time">
                <xsl:value-of select="xunit:junit-time(@time)"/>
            </xsl:attribute>
            <xsl:apply-templates select="testcase"/>
        </testsuite>
    </xsl:template>
    <xsl:template match="//testcase">
        <testcase>
            <xsl:choose>
                <xsl:when test="@value_param">
                    <xsl:attribute name="name">
                        <xsl:value-of select="@name"/> (<xsl:value-of select="@value_param"/>)
                    </xsl:attribute>
                </xsl:when>
                <xsl:otherwise>
                    <xsl:attribute name="name">
                        <xsl:value-of select="@name"/>
                    </xsl:attribute>
                </xsl:otherwise>
            </xsl:choose>
            <xsl:attribute name="time">
                <xsl:value-of select="xunit:junit-time(@time)"/>
            </xsl:attribute>
            <xsl:attribute name="classname">
                <xsl:value-of select="@classname"/>
            </xsl:attribute>
            <xsl:if test="@status = 'notrun'">
                <skipped/>
            </xsl:if>
            <xsl:if test="skipped">
                <skipped/>
            </xsl:if>
            <xsl:if test="failure">
                <failure>
                    <xsl:for-each select="failure">
                        <xsl:if test="not(position()=1)">
                            <xsl:text>&#xa;&#xa;</xsl:text>
                        </xsl:if>
                        <xsl:value-of select="@message"/>
                    </xsl:for-each>
                </failure>
                <system-out>
                    <xsl:for-each select="failure">
                        <xsl:if test="not(position()=1)">
                            <xsl:text>&#xa;&#xa;</xsl:text>
                        </xsl:if>
                        <xsl:value-of select="."/>
                    </xsl:for-each>
                </system-out>
                <system-err>
                    <xsl:for-each select="system-out">
                        <xsl:if test="not(position()=1)">
                            <xsl:text>&#xa;&#xa;</xsl:text>
                        </xsl:if>
                        <xsl:value-of select="."/>
                    </xsl:for-each>
                </system-err>

            </xsl:if>
	        <xsl:if test="error">
                <failure>
                    <xsl:for-each select="error">
                        <xsl:if test="not(position()=1)">
                            <xsl:text>&#xa;&#xa;</xsl:text>
                        </xsl:if>
                        <xsl:value-of select="@message"/>
                    </xsl:for-each>
                </failure>
                <system-out>
                    <xsl:for-each select="error">
                        <xsl:if test="not(position()=1)">
                            <xsl:text>&#xa;&#xa;</xsl:text>
                        </xsl:if>
                        <xsl:value-of select="."/>
                    </xsl:for-each>
                </system-out>
            </xsl:if>


            <system-out>
                <xsl:for-each select="system-out">
                    <xsl:if test="not(position()=1)">
                        <xsl:text>&#xa;&#xa;</xsl:text>
                    </xsl:if>
                    <xsl:value-of select="."/>
                </xsl:for-each>
            </system-out>
            <system-err>
                <xsl:for-each select="system-err">
                    <xsl:if test="not(position()=1)">
                        <xsl:text>&#xa;&#xa;</xsl:text>
                    </xsl:if>
                    <xsl:value-of select="."/>
                </xsl:for-each>
            </system-err>


        </testcase>
    </xsl:template>
</xsl:stylesheet>

